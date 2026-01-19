#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import csv
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, RMSprop
from transformers import BertTokenizer

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from fusions.common_fusions import (
    Concat,
    ConcatEarly,
    CrossAttentionConcatFusion,
    CrossAttentionFusion,
    EarlyFusionTransformer,
    HierarchicalAttentionMultiToOne,
    HierarchicalAttentionOneToMulti,
    LateFusionTransformer,
    TensorFusion,
)
from fusions.late_fusion import MultimodalLateFusionClf
from fusions.tmc import TMC
from get_data import get_loader
from encoders.bert import BertEncoder
from encoders.image import ImageEncoder
from training_structures.Supervised_Learning_V2 import test, train
import copy

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Args:
    bert_model: str = "bert-base-uncased"
    hidden_sz: int = 768
    num_image_embeds: int = 1
    img_embed_pool_type: str = "avg"
    img_hidden_sz: int = 2048


args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

tokenizer = BertTokenizer.from_pretrained("../../bert-base-uncased")

def get_head(num_classes: int, decision: bool = False):
    if decision:
        return nn.ModuleList([nn.LazyLinear(num_classes), nn.LazyLinear(num_classes)])
    return nn.LazyLinear(num_classes)


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path, _use_new_zipfile_serialization=True)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path, map_location=device):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    logger.info(f"Loaded checkpoint from {path}")
    return ckpt


# ----------------------- Optuna objective --------------------
def objective(
    trial: optuna.Trial,
    fusion,
    train_loader,
    val_loader,
    head,
    fusion_type: str,
    seed: int,
):
    """Optuna objective: maximise validation accuracy."""
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    freeze_encoders = trial.suggest_categorical("freeze_encoders", [True,False])
    optim_name = trial.suggest_categorical("optimtype", ["AdamW", "RMSprop", "Adam"])

    optim_cls = {"AdamW": AdamW, "RMSprop": RMSprop, "Adam": Adam}[optim_name]

    # Build fresh encoders for each trial to avoid global side-effects
    image_encoder = ImageEncoder(args).to(device)
    text_encoder = BertEncoder(args).to(device)

    fusion_model = copy.deepcopy(fusion).to(device)
    head_model = copy.deepcopy(head).to(device)

    trial_output = OUTPUT_DIR / f"{fusion.__class__.__name__}_seed{seed}_trial{trial.number}"
    trial_output.mkdir(exist_ok=True)
    best_model_path = trial_output / "best.pt"

    best_acc = train(
        encoders=[image_encoder, text_encoder],
        fusion=fusion_model,
        head=head_model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        total_epochs=100,
        optimtype=optim_cls,
        lr=lr,
        weight_decay=weight_decay,
        task="classification",
        save=str(best_model_path),
        tokenizer=tokenizer,
        freeze_encoders=freeze_encoders,
        fusion_type=fusion_type,
        early_stop=True,
    )

    trial.set_user_attr("best_model_path", str(best_model_path))
    return best_acc


# ----------------------- main pipeline -----------------------
def main():
    """Run full benchmark loop for all fusion methods and seeds."""
    train_loader, val_loader, test_loader, num_classes = get_loader()


    fusion_hub = {
        "Concat": Concat(),
        "TensorFusion": TensorFusion(),
        "ConcatEarly": ConcatEarly(),
        "LateFusionTransformer": LateFusionTransformer(),
        "HierarchicalAttentionOneToMulti": HierarchicalAttentionOneToMulti([args.img_hidden_sz, args.hidden_sz]),
        "CrossAttentionFusion": CrossAttentionFusion([args.img_hidden_sz, args.hidden_sz]),
        "CrossAttentionConcatFusion": CrossAttentionConcatFusion([args.img_hidden_sz, args.hidden_sz]),
        "EarlyFusionTransformer": EarlyFusionTransformer(args.img_hidden_sz + args.hidden_sz),
        "TMC": TMC(num_classes),
        "LateFusion": MultimodalLateFusionClf(),
        "HierarchicalAttentionMultiToOne": HierarchicalAttentionMultiToOne([args.img_hidden_sz, args.hidden_sz]),
    }

    for seed in [1, 2, 3]:
        set_seed(seed)
        logger.info(f"========== SEED {seed} ==========")

        for fusion_name, fusion in fusion_hub.items():
            logger.info(f"--- {fusion_name} ---")

            # Decide feature- vs decision-level fusion
            if fusion_name in {"TMC", "LateFusion"}:
                fusion_type = "decision"
                head = get_head(num_classes, decision=True)
            else:
                fusion_type = "feature"
                head = get_head(num_classes, decision=False)

            study = optuna.create_study(
                direction="maximize",
                study_name=f"{fusion_name}_seed{seed}",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
            study.optimize(
                lambda trial: objective(
                    trial,
                    fusion,
                    train_loader,
                    val_loader,
                    head,
                    fusion_type,
                    seed,
                ),
                n_trials=20,
                show_progress_bar=True,
            )

            logger.info(f"[{fusion_name}] Best params: {study.best_params}")
            logger.info(f"[{fusion_name}] Best val acc: {study.best_value:.4f}")

            # ----------- test phase -----------
            best_path = study.best_trial.user_attrs["best_model_path"]
            ckpt = load_checkpoint(Path(best_path))
            model = ckpt if isinstance(ckpt, nn.Module) else ckpt["state_dict"]
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            test_acc = test(model, test_loader, criterion=criterion, no_robust=True)
            logger.info(f"[{fusion_name}] Test acc: {test_acc:.4f}")

            # ----------- save results -----------
            study.trials_dataframe().to_csv(
                OUTPUT_DIR / f"{fusion_name}_seed{seed}_trials.csv", index=False
            )
            with open(OUTPUT_DIR / f"test_results_seed{seed}.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([fusion_name, f"{test_acc:.4f}"])


if __name__ == "__main__":
    main()