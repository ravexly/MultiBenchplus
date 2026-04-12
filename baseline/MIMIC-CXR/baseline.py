from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline._shared.experiment_runner import (
    build_classification_fusion_methods,
    build_classification_head,
    run_fusion_experiment,
    setup_baseline_runtime,
)
from get_data import get_loader
from encoders.bert import BertEncoder
from encoders.image import ImageEncoder
from training_structures.Supervised_Learning_V2 import test, train

DATASET_NAME = 'MIMIC-CXR'
OUTPUT_DIR = Path(__file__).resolve().parent
ROOT, SOURCE_DATASET_DIR, OUTPUT_DIR = setup_baseline_runtime(DATASET_NAME, OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = (1, 2, 3)
N_TRIALS = 5
TOTAL_EPOCHS = 100
DECISION_FUSIONS = frozenset({'TMC', 'LateFusion'})
TOKENIZER = BertTokenizer.from_pretrained('../../bert-base-uncased')


class Args:
    bert_model = 'bert-base-uncased'
    hidden_sz = 768
    num_image_embeds = 1
    img_embed_pool_type = 'avg'
    img_hidden_sz = 2048
    n_classes = 3


ARGS = Args()


def build_encoders() -> list[nn.Module]:
    return [
        ImageEncoder(ARGS).to(device),
        BertEncoder(ARGS).to(device),
    ]


def build_fusion_methods(num_classes: int) -> dict[str, nn.Module]:
    return build_classification_fusion_methods(
        num_classes,
        [2048, 768],
        include_concat=False,
        include_tensor=False,
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
    train_loader, val_loader, test_loader, num_classes = get_loader()
    run_fusion_experiment(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        fusion_methods=build_fusion_methods(num_classes),
        make_encoders=build_encoders,
        build_head=build_head,
        seeds=SEEDS,
        n_trials=N_TRIALS,
        total_epochs=TOTAL_EPOCHS,
        direction='maximize',
        task='multilabel',
        criterion_factory=nn.BCEWithLogitsLoss,
        decision_fusions=DECISION_FUSIONS,
        tokenizer=TOKENIZER,
        output_dir=OUTPUT_DIR,
        train_fn=train,
        test_fn=test,
    )


if __name__ == '__main__':
    main()
