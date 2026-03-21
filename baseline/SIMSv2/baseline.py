from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline._shared.experiment_runner import (
    build_classification_fusion_methods,
    build_regression_head,
    run_fusion_experiment,
    setup_baseline_runtime,
)
from get_data import get_loader
from training_structures.Supervised_Learning_V2 import test, train
from unimodals.common_models import GRU

DATASET_NAME = 'SIMSv2'
OUTPUT_DIR = Path(__file__).resolve().parent
ROOT, SOURCE_DATASET_DIR, OUTPUT_DIR = setup_baseline_runtime(DATASET_NAME, OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = (1, 2, 3)
N_TRIALS = 10
TOTAL_EPOCHS = 100
DECISION_FUSIONS = frozenset({'LateFusion'})


def build_encoders() -> list[nn.Module]:
    return [
        GRU(50, 128, last_only=True).to(device),
        GRU(25, 128, last_only=True).to(device),
        nn.Sequential(nn.Linear(177, 128), GRU(128, 128, last_only=True)).to(device),
    ]


def build_fusion_methods(num_classes: int) -> dict[str, nn.Module]:
    del num_classes
    return build_classification_fusion_methods(
        1,
        [128, 128, 128],
        include_cross_attention=False,
        include_multi_cross_attention=True,
        include_tmc=False,
    )


def build_head(num_classes: int, fusion_name: str):
    del num_classes
    if fusion_name in DECISION_FUSIONS:
        return [build_regression_head(1, device=device) for _ in range(3)]
    return build_regression_head(1, device=device)


def main() -> None:
    train_loader, val_loader, test_loader = get_loader()
    run_fusion_experiment(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        num_classes=1,
        fusion_methods=build_fusion_methods(1),
        make_encoders=build_encoders,
        build_head=build_head,
        seeds=SEEDS,
        n_trials=N_TRIALS,
        total_epochs=TOTAL_EPOCHS,
        direction='minimize',
        task='regression',
        criterion_factory=nn.MSELoss,
        decision_fusions=DECISION_FUSIONS,
        tokenizer=None,
        output_dir=OUTPUT_DIR,
        train_fn=train,
        test_fn=test,
    )


if __name__ == '__main__':
    main()
