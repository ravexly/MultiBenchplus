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
    build_classification_head,
    run_fusion_experiment,
    setup_baseline_runtime,
)
from encoders.compact_branch_cnn import CompactBranchCNNEncoder
from get_data import get_loader
from training_structures.Supervised_Learning_V2 import test, train

DATASET_NAME = 'augsburg'
OUTPUT_DIR = Path(__file__).resolve().parent
ROOT, SOURCE_DATASET_DIR, OUTPUT_DIR = setup_baseline_runtime(DATASET_NAME, OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = (1, 2, 3)
N_TRIALS = 10
TOTAL_EPOCHS = 50
DECISION_FUSIONS = frozenset({'TMC', 'LateFusion'})


def build_encoders() -> list[nn.Module]:
    return [
        CompactBranchCNNEncoder(64, 16).to(device),
        CompactBranchCNNEncoder(8, 16).to(device),
    ]


def build_fusion_methods(num_classes: int) -> dict[str, nn.Module]:
    return build_classification_fusion_methods(
        num_classes,
        [16, 16],
        include_multi_cross_attention=False,
        include_tmc=True,
    )


def build_head(num_classes: int, fusion_name: str):
    return build_classification_head(
        num_classes,
        decision=fusion_name in DECISION_FUSIONS,
        num_decision_heads=2,
        device=device,
    )


def main() -> None:
    train_loader, test_loader, num_classes = get_loader()
    run_fusion_experiment(
        train_loader=train_loader,
        valid_loader=test_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        fusion_methods=build_fusion_methods(num_classes),
        make_encoders=build_encoders,
        build_head=build_head,
        seeds=SEEDS,
        n_trials=N_TRIALS,
        total_epochs=TOTAL_EPOCHS,
        direction='maximize',
        task='classification',
        criterion_factory=nn.CrossEntropyLoss,
        decision_fusions=DECISION_FUSIONS,
        tokenizer=None,
        output_dir=OUTPUT_DIR,
        train_fn=train,
        test_fn=test,
    )


if __name__ == '__main__':
    main()
