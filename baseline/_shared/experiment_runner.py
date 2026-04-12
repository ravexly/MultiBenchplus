from __future__ import annotations

import copy
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, RMSprop
from fusions.common_fusions import (
    Concat,
    ConcatEarly,
    CrossAttentionConcatFusion,
    CrossAttentionFusion,
    EarlyFusionTransformer,
    HierarchicalAttentionMultiToOne,
    HierarchicalAttentionOneToMulti,
    LateFusionTransformer,
    MultiModalCrossAttentionConcatFusion,
    MultiModalCrossAttentionFusion,
    TensorFusion,
)
from fusions.late_fusion import MultimodalLateFusionClf
from fusions.tmc import TMC

ROOT = Path(__file__).resolve().parents[2]

for _path in (ROOT,):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import optuna


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_classification_head(
    num_classes: int,
    *,
    decision: bool = False,
    num_decision_heads: int = 2,
    device: torch.device | None = None,
) -> nn.Module | list[nn.Module]:

    if decision:
        return [nn.LazyLinear(num_classes).to(device) for _ in range(num_decision_heads)]
    return nn.LazyLinear(num_classes).to(device)


def build_regression_head(
    output_dim: int = 1,
    *,
    device: torch.device | None = None,
) -> nn.Module:
    return nn.Sequential(nn.LazyLinear(output_dim)).to(device)


def build_classification_fusion_methods(
    num_classes: int,
    dims: Sequence[int],
    *,
    include_concat: bool = True,
    include_tensor: bool = True,
    include_cross_attention: bool = True,
    include_multi_cross_attention: bool = False,
    include_late_fusion: bool = True,
    include_tmc: bool = True,
    include_transformer: bool = True,
    include_hierarchical: bool = True,
) -> dict[str, nn.Module]:
    fusion_methods: dict[str, nn.Module] = {}

    if include_concat:
        fusion_methods["Concat"] = Concat()
    if include_tensor:
        fusion_methods["TensorFusion"] = TensorFusion()
    if include_concat:
        fusion_methods["ConcatEarly"] = ConcatEarly()
    if include_transformer:
        fusion_methods["LateFusionTransformer"] = LateFusionTransformer()
    if include_hierarchical:
        fusion_methods["HierarchicalAttentionOneToMulti"] = HierarchicalAttentionOneToMulti(list(dims))
    if include_cross_attention:
        fusion_methods["CrossAttentionFusion"] = CrossAttentionFusion(list(dims))
        fusion_methods["CrossAttentionConcatFusion"] = CrossAttentionConcatFusion(list(dims))
    if include_multi_cross_attention:
        fusion_methods["MultiModalCrossAttentionFusion"] = MultiModalCrossAttentionFusion(list(dims))
        fusion_methods["MultiModalCrossAttentionConcatFusion"] = MultiModalCrossAttentionConcatFusion(list(dims))

    fusion_methods["EarlyFusionTransformer"] = EarlyFusionTransformer(sum(dims))
    if include_tmc:
        fusion_methods["TMC"] = TMC(num_classes)
    if include_late_fusion:
        fusion_methods["LateFusion"] = MultimodalLateFusionClf()
    if include_hierarchical:
        fusion_methods["HierarchicalAttentionMultiToOne"] = HierarchicalAttentionMultiToOne(list(dims))

    return fusion_methods


def build_optimizer_map() -> dict[str, type[torch.optim.Optimizer]]:
    return {
        "AdamW": AdamW,
        "RMSprop": RMSprop,
        "Adam": Adam,
    }


def _save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    torch.save({"model": model, "metadata": metadata}, path)


def _load_checkpoint_model(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, weights_only=False)
    model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(model, nn.Module):
        raise TypeError(f"Unexpected checkpoint format at {path}")
    return model.to(device)


def run_fusion_experiment(
    *,
    train_loader,
    valid_loader,
    test_loader,
    num_classes: int,
    fusion_methods: dict[str, nn.Module],
    make_encoders: Callable[[], list[nn.Module]],
    build_head: Callable[[int, str], nn.Module | list[nn.Module]],
    seeds: Sequence[int],
    n_trials: int,
    total_epochs: int,
    direction: str,
    task: str,
    criterion_factory: Callable[[], nn.Module],
    optimizer_choices: Sequence[str] = ("AdamW", "RMSprop", "Adam"),
    lr_range: tuple[float, float] = (1e-5, 1e-3),
    weight_decay_range: tuple[float, float] = (1e-6, 1e-2),
    freeze_choices: Sequence[bool] = (False,),
    decision_fusions: frozenset[str] = frozenset(),
    tokenizer=None,
    output_dir: Path | None = None,
    train_fn=None,
    test_fn=None,
    eval_metric: str = "accuracy",
    auroc_average: str = "macro",
    auroc_multi_class: str = "ovr",
) -> None:
    train_fn = train_fn or _default_train
    test_fn = test_fn or _default_test
    output_dir = output_dir or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_map = build_optimizer_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_validation_path = output_dir / "best_validation.pt"
    best_overall_path = output_dir / "best_overall.pt"

    for seed in seeds:
        seed_everything(seed)
        test_results: dict[str, float] = {}
        for fusion_name, fusion in fusion_methods.items():
            best_value = -float("inf") if direction == "maximize" else float("inf")
            head_template = build_head(num_classes, fusion_name)
            fusion_type = "decision" if fusion_name in decision_fusions else "feature"

            def objective(trial: optuna.Trial) -> float:
                nonlocal best_value

                lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
                weight_decay = trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1], log=True)
                freeze_encoders = trial.suggest_categorical("freeze_encoders", list(freeze_choices))
                optimizer_name = trial.suggest_categorical("optimtype", list(optimizer_choices))
                checkpoint_metadata = {
                    "dataset": output_dir.name,
                    "fusion": fusion_name,
                    "seed": seed,
                    "trial": trial.number,
                    "score": None,
                    "task": task,
                    "direction": direction,
                    "eval_metric": eval_metric,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "freeze_encoders": freeze_encoders,
                    "optimizer": optimizer_name,
                }

                score = train_fn(
                    encoders=make_encoders(),
                    fusion=copy.deepcopy(fusion),
                    head=copy.deepcopy(head_template),
                    train_dataloader=train_loader,
                    valid_dataloader=valid_loader,
                    total_epochs=total_epochs,
                    optimtype=optimizer_map[optimizer_name],
                    lr=lr,
                    weight_decay=weight_decay,
                    task=task,
                    save=str(best_validation_path),
                    tokenizer=tokenizer,
                    freeze_encoders=freeze_encoders,
                    fusion_type=fusion_type,
                    early_stop=True,
                    objective=criterion_factory(),
                    eval_metric=eval_metric,
                    auroc_average=auroc_average,
                    auroc_multi_class=auroc_multi_class,
                )

                checkpoint_metadata["score"] = float(score)
                validation_model = _load_checkpoint_model(best_validation_path, device)
                _save_checkpoint(best_validation_path, validation_model, checkpoint_metadata)

                better = score > best_value if direction == "maximize" else score < best_value
                if better:
                    best_value = score
                    _save_checkpoint(best_overall_path, validation_model, checkpoint_metadata)
                return score

            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction=direction, study_name=f"{fusion_name}", sampler=sampler)
            study.optimize(objective, n_trials=n_trials)

            model = _load_checkpoint_model(best_overall_path, device)
            test_results[fusion_name] = test_fn(
                model,
                test_loader,
                criterion=criterion_factory(),
                no_robust=True,
                task=task,
                eval_metric=eval_metric,
                auroc_average=auroc_average,
                auroc_multi_class=auroc_multi_class,
            )

            study.trials_dataframe().to_csv(output_dir / f"{fusion_name}_{seed}_trials.csv", index=False)
            with open(output_dir / f"test_results_{seed}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(test_results.keys())
                writer.writerow(test_results.values())


def _default_train(*args, **kwargs):
    from training_structures.Supervised_Learning_V2 import train

    return train(*args, **kwargs)


def _default_test(*args, **kwargs):
    from training_structures.Supervised_Learning_V2 import test

    return test(*args, **kwargs)


def setup_baseline_runtime(dataset_name: str, output_dir: Path) -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    source_dataset_dir = root / "output" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    for path in (root, root / "src", source_dataset_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root, source_dataset_dir, output_dir
