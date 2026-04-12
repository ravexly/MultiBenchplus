import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline._shared.experiment_runner import (
    build_classification_fusion_methods,
    build_classification_head,
    run_fusion_experiment,
    seed_everything,
    setup_baseline_runtime,
)

OUTPUT_DIR = Path(__file__).resolve().parent
ROOT, SOURCE_DATASET_DIR, OUTPUT_DIR = setup_baseline_runtime("mvsa_single", OUTPUT_DIR)

import os
import sys

import torch
import optuna
import copy
import torch.nn as nn
from torch.optim import AdamW, RMSprop, Adam
from torch.nn import Sequential, LazyConv2d, ReLU, LazyBatchNorm2d, LazyLinear

from get_data import get_loader
from training_structures.Supervised_Learning_V2 import train, test
from fusions.common_fusions import (
    Concat, TensorFusion, ConcatEarly,
    EarlyFusionTransformer, CrossAttentionFusion,
    CrossAttentionConcatFusion, HierarchicalAttentionMultiToOne,
    HierarchicalAttentionOneToMulti, LateFusionTransformer
)
from fusions.late_fusion import MultimodalLateFusionClf
from fusions.tmc import TMC
from encoders.image import ImageEncoder
from encoders.bert import BertEncoder
from transformers import BertTokenizer
import random
import numpy as np
from optuna.samplers import TPESampler
import csv
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
modality_num = 2
class Args:
    bert_model = "bert-base-uncased"
    hidden_sz = 768
    num_image_embeds = 1
    img_embed_pool_type = "avg"
    img_hidden_sz = 2048
    n_classes =3

args = Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  BertTokenizer.from_pretrained("../../bert-base-uncased")

def get_head(num_classes, decision=False):
    if decision:
        return [LazyLinear(num_classes).cuda(), LazyLinear(num_classes).cuda()]
    else:
        return LazyLinear(num_classes).cuda()


def build_encoders():
    optim_map = {'AdamW': AdamW, 'RMSprop': RMSprop, 'Adam': Adam}
    image_encoder = ImageEncoder(args).to(device)
    text_encoder = BertEncoder(args).to(device)
    return [image_encoder, text_encoder]


def build_fusion_methods(num_classes: int) -> dict[str, nn.Module]:
    return build_classification_fusion_methods(
        num_classes,
        [2048, 768],
        
    )

SEEDS = (1, 2, 3)
N_TRIALS = 10
TOTAL_EPOCHS = 100
DECISION_FUSIONS = frozenset({"TMC", "LateFusion"})


def build_head(num_classes: int, fusion_name: str):
    return build_classification_head(
        num_classes,
        decision=fusion_name in DECISION_FUSIONS,
        num_decision_heads=2,
        device=device,
    )


def main() -> None:
    train_loader, val_loader, test_loader, num_classes = get_loader()
    fusion_methods = build_fusion_methods(num_classes)
    run_fusion_experiment(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        fusion_methods=fusion_methods,
        make_encoders=build_encoders,
        build_head=build_head,
        seeds=SEEDS,
        n_trials=N_TRIALS,
        total_epochs=TOTAL_EPOCHS,
        direction="maximize",
        task="classification",
        criterion_factory=nn.CrossEntropyLoss,
        decision_fusions=DECISION_FUSIONS,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        train_fn=train,
        test_fn=test,
    )


if __name__ == "__main__":
    main()
