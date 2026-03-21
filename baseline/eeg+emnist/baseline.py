from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline._shared.experiment_runner import (
    build_classification_head,
    run_fusion_experiment,
    setup_baseline_runtime,
)
from encoders.sequence_lstm import SequenceLSTMEncoder
from encoders.temporal_image_sequence import TemporalImageSequenceEncoder
from fusions.common_fusions import Concat, ConcatEarly, CrossAttentionConcatFusion, CrossAttentionFusion, EarlyFusionTransformer, HierarchicalAttentionMultiToOne, HierarchicalAttentionOneToMulti, LateFusionTransformer, TensorFusion
from fusions.late_fusion import MultimodalLateFusionClf
from fusions.tmc import TMC
DATASET_NAME = "eeg+emnist"
OUTPUT_DIR = Path(__file__).resolve().parent
ROOT, SOURCE_DATASET_DIR, OUTPUT_DIR = setup_baseline_runtime(DATASET_NAME, OUTPUT_DIR)

from get_data import get_loader
from training_structures.Supervised_Learning_V2 import test, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (1, 2, 3)
N_TRIALS = 10
TOTAL_EPOCHS = 100
DECISION_FUSIONS = frozenset({"TMC", "LateFusion"})


def build_encoders() -> list[nn.Module]:
    return [SequenceLSTMEncoder(192, 128, 2).to(device), TemporalImageSequenceEncoder().to(device)]


def build_fusion_methods(num_classes: int) -> dict[str, nn.Module]:
    return {
        "Concat": Concat(),
        "TensorFusion": TensorFusion(),
        "ConcatEarly": ConcatEarly(),
        "LateFusionTransformer": LateFusionTransformer(),
        "HierarchicalAttentionOneToMulti": HierarchicalAttentionOneToMulti([128, 128]),
        "CrossAttentionFusion": CrossAttentionFusion([128, 128]),
        "CrossAttentionConcatFusion": CrossAttentionConcatFusion([128, 128]),
        "EarlyFusionTransformer": EarlyFusionTransformer(256),
        "TMC": TMC(num_classes),
        "LateFusion": MultimodalLateFusionClf(),
        "HierarchicalAttentionMultiToOne": HierarchicalAttentionMultiToOne([128, 128]),
    }


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
        optimizer_choices=("AdamW", "RMSprop", "Adam"),
        lr_range=(1e-5, 1e-3),
        weight_decay_range=(1e-6, 1e-2),
        freeze_choices=(False,),
        decision_fusions=DECISION_FUSIONS,
        tokenizer=None,
        output_dir=OUTPUT_DIR,
        train_fn=train,
        test_fn=test,
    )


if __name__ == "__main__":
    main()
