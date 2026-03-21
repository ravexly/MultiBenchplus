from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline"
EXPER_DIR = ROOT / "exper"

baseline_dir_str = str(BASELINE_DIR)
while baseline_dir_str in sys.path:
    sys.path.remove(baseline_dir_str)

for path in (ROOT, ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import optuna  # type: ignore  # noqa: E402

if baseline_dir_str not in sys.path:
    sys.path.insert(1, baseline_dir_str)

if "memory_profiler" not in sys.modules:
    memory_profiler_stub = types.ModuleType("memory_profiler")

    def memory_usage(*args, **kwargs):
        del args, kwargs
        return 0.0, None

    memory_profiler_stub.memory_usage = memory_usage
    sys.modules["memory_profiler"] = memory_profiler_stub


if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    tokenization_base_stub = types.ModuleType("transformers.tokenization_utils_base")

    class BatchEncoding(dict):
        pass

    class BertTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def _encode_text(self, text, max_length: int = 32):
            tokens = str(text).split()
            ids = [min(abs(hash(token)) % 30522, 30521) for token in tokens[:max_length]]
            if not ids:
                ids = [0]
            attention = [1] * len(ids)
            if len(ids) < max_length:
                pad = max_length - len(ids)
                ids = ids + [0] * pad
                attention = attention + [0] * pad
            return ids, attention

        def batch_encode_plus(self, texts, padding=True, truncation=True, return_tensors="pt", max_length=32, **kwargs):
            del padding, truncation, kwargs
            encoded = [self._encode_text(text, max_length=max_length) for text in texts]
            input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
            attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.long)
            token_type_ids = torch.zeros_like(input_ids)
            return BatchEncoding(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )

        __call__ = batch_encode_plus

    transformers_stub.BertTokenizer = BertTokenizer
    tokenization_base_stub.BatchEncoding = BatchEncoding
    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.tokenization_utils_base"] = tokenization_base_stub


if "pytorch_pretrained_bert" not in sys.modules:
    bert_package_stub = types.ModuleType("pytorch_pretrained_bert")
    bert_modeling_stub = types.ModuleType("pytorch_pretrained_bert.modeling")

    class BertModel(nn.Module):
        def __init__(self, hidden_size: int = 768) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.embeddings = nn.Embedding(30522, hidden_size)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def init_bert_weights(self, module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):
            del token_type_ids, attention_mask, output_all_encoded_layers
            x = self.embeddings(input_ids)
            pooled = x.mean(dim=1)
            return None, pooled

    bert_modeling_stub.BertModel = BertModel
    bert_package_stub.modeling = bert_modeling_stub
    sys.modules["pytorch_pretrained_bert"] = bert_package_stub
    sys.modules["pytorch_pretrained_bert.modeling"] = bert_modeling_stub


def discover_baselines() -> list[Path]:
    return sorted(path for path in BASELINE_DIR.glob("*/baseline.py") if path.parent.name != "_shared")


def load_module(module_path: Path):
    module_name = f"smoke_{module_path.parent.name.replace('+', '_').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_runtime_paths(module_path: Path) -> None:
    dataset_dir = EXPER_DIR / module_path.parent.name
    for path in (dataset_dir, ROOT / "src", ROOT, BASELINE_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def patch_train_function(train_fn):
    signature = inspect.signature(train_fn)

    def patched_train(*args, **kwargs):
        for name in ("total_epochs", "epoch", "epochs"):
            if name in signature.parameters:
                kwargs[name] = 1
        if "track_complexity" in signature.parameters:
            kwargs["track_complexity"] = False
        if "early_stop" in signature.parameters:
            kwargs["early_stop"] = True
        return train_fn(*args, **kwargs)

    return patched_train


def patch_baseline_module(module) -> None:
    if hasattr(module, "TOTAL_EPOCHS"):
        module.TOTAL_EPOCHS = 1
    if hasattr(module, "N_TRIALS"):
        module.N_TRIALS = 1
    if hasattr(module, "track_complexity"):
        module.track_complexity = False
    if hasattr(module, "SEEDS"):
        module.SEEDS = (1,)
    if hasattr(module, "SEED"):
        module.SEED = 1

    if hasattr(module, "train") and callable(module.train):
        module.train = patch_train_function(module.train)

    if hasattr(module, "build_fusion_methods") and callable(module.build_fusion_methods):
        original_build_fusion_methods = module.build_fusion_methods

        def patched_build_fusion_methods(*args, **kwargs):
            methods = original_build_fusion_methods(*args, **kwargs)
            if isinstance(methods, dict) and len(methods) > 1:
                first_key = next(iter(methods))
                return {first_key: methods[first_key]}
            return methods

        module.build_fusion_methods = patched_build_fusion_methods


def run_single_baseline(module_path: Path) -> int:
    prepare_runtime_paths(module_path)
    module = load_module(module_path)
    patch_baseline_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{module_path} has no main()")
    module.main()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset folder names to run. Defaults to all baseline folders.",
    )
    parser.add_argument(
        "--start-from",
        help="Optional dataset folder name to resume from, inclusive.",
    )
    args = parser.parse_args()

    baselines = discover_baselines()
    if args.datasets:
        wanted = set(args.datasets)
        baselines = [path for path in baselines if path.parent.name in wanted]
    if args.start_from:
        start_names = [path.parent.name for path in baselines]
        if args.start_from not in start_names:
            raise SystemExit(f"Unknown dataset for --start-from: {args.start_from}")
        baselines = baselines[start_names.index(args.start_from)+1 :]

    failures: list[tuple[str, int]] = []
    for baseline_path in baselines:
        dataset_name = baseline_path.parent.name
        print(f"\n=== Smoke test: {dataset_name} ===")
        result = subprocess.run(
            [sys.executable, __file__, "--single", str(baseline_path)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            failures.append((dataset_name, result.returncode))

    if failures:
        print("\nFailures:")
        for dataset_name, code in failures:
            print(f"{dataset_name}: exit code {code}")
        return 1
    return 0


if __name__ == "__main__":
    if "--single" in sys.argv:
        single_index = sys.argv.index("--single")
        module_path = Path(sys.argv[single_index + 1])
        sys.argv = [sys.argv[0]]
        raise SystemExit(run_single_baseline(module_path))
    raise SystemExit(main())
