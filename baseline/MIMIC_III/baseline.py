from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline._shared.experiment_runner import (
    build_regression_head,
    run_fusion_experiment,
    setup_baseline_runtime,
)
from encoders.sequence_transformer import SequenceTransformerEncoder
from fusions.common_fusions import Concat, ConcatEarly, CrossAttentionConcatFusion, CrossAttentionFusion, EarlyFusionTransformer, HierarchicalAttentionMultiToOne, HierarchicalAttentionOneToMulti, LateFusionTransformer, MultiModalCrossAttentionConcatFusion, MultiModalCrossAttentionFusion, TensorFusion
from fusions.late_fusion import MultimodalLateFusionClf
DATASET_NAME = "MIMIC_III"
OUTPUT_DIR = Path(__file__).resolve().parent
ROOT, SOURCE_DATASET_DIR, OUTPUT_DIR = setup_baseline_runtime(DATASET_NAME, OUTPUT_DIR)

from get_data import get_loader
from training_structures.Supervised_Learning_V2 import test, train
from unimodals.common_models import Identity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (1, 2, 3)
N_TRIALS = 10
TOTAL_EPOCHS = 100
DECISION_FUSIONS = frozenset({"LateFusion"})


def build_encoders() -> list[nn.Module]:
    return [Identity().to(device), SequenceTransformerEncoder(12, 32).to(device)]


def build_fusion_methods(num_classes: int) -> dict[str, nn.Module]:
    del num_classes
    return {
        "Concat": Concat(),
        "TensorFusion": TensorFusion(),
        "ConcatEarly": ConcatEarly(),
        "LateFusionTransformer": LateFusionTransformer(),
        "HierarchicalAttentionOneToMulti": HierarchicalAttentionOneToMulti([5, 32]),
        "MultiModalCrossAttentionFusion": MultiModalCrossAttentionFusion([5, 32]),
        "MultiModalCrossAttentionConcatFusion": MultiModalCrossAttentionConcatFusion([5, 32]),
        "EarlyFusionTransformer": EarlyFusionTransformer(37),
        "LateFusion": MultimodalLateFusionClf(),
        "HierarchicalAttentionMultiToOne": HierarchicalAttentionMultiToOne([5, 32]),
    }


def build_head(num_classes: int, fusion_name: str):
    del num_classes
    if fusion_name in DECISION_FUSIONS:
        return [build_regression_head(1, device=device), build_regression_head(1, device=device)]
    return build_regression_head(1, device=device)


def main() -> None:
    train_loader, val_loader, test_loader = get_loader()
    fusion_methods = build_fusion_methods(1)
    run_fusion_experiment(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        num_classes=1,
        fusion_methods=fusion_methods,
        make_encoders=build_encoders,
        build_head=build_head,
        seeds=SEEDS,
        n_trials=N_TRIALS,
        total_epochs=TOTAL_EPOCHS,
        direction="minimize",
        task="regression",
        criterion_factory=nn.MSELoss,
        optimizer_choices=("AdamW", "RMSprop", "Adam"),
        lr_range=(1e-5, 1e-3),
        weight_decay_range=(1e-6, 1e-3),
        freeze_choices=(False,),
        decision_fusions=DECISION_FUSIONS,
        tokenizer=None,
        output_dir=OUTPUT_DIR,
        train_fn=train,
        test_fn=test,
    )


if __name__ == "__main__":
    main()
