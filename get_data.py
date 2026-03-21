from __future__ import annotations

import importlib.util
import inspect
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent


def _infer_dataset_name() -> str:
    frame = inspect.currentframe()
    try:
        while frame is not None:
            globals_dict = frame.f_globals
            dataset_name = globals_dict.get("DATASET_NAME")
            if isinstance(dataset_name, str) and dataset_name:
                return dataset_name

            file_name = globals_dict.get("__file__")
            if file_name:
                path = Path(file_name).resolve()
                if path.name == "baseline.py":
                    parent_name = path.parent.name
                    if parent_name not in {"baseline", "_shared", "exper"}:
                        return parent_name

            frame = frame.f_back
    finally:
        del frame

    raise RuntimeError("Unable to infer dataset name for get_loader import")


def _module_name(dataset_name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in dataset_name)
    return f"_multibenchplus_dataset_{safe_name}_get_data"


@lru_cache(maxsize=None)
def _load_dataset_module(dataset_name: str) -> ModuleType:
    dataset_dir = ROOT / "dataset" / dataset_name
    dataset_file = dataset_dir / "get_data.py"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Missing dataset loader: {dataset_file}")

    dataset_dir_str = str(dataset_dir)
    if dataset_dir_str not in sys.path:
        sys.path.insert(0, dataset_dir_str)

    module_name = _module_name(dataset_name)
    spec = importlib.util.spec_from_file_location(module_name, dataset_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module from {dataset_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_loader(*args: Any, **kwargs: Any):
    dataset_name = _infer_dataset_name()
    module = _load_dataset_module(dataset_name)
    loader = getattr(module, "get_loader", None)
    if loader is None or not callable(loader):
        raise AttributeError(f"{module.__file__} does not define a callable get_loader")
    return loader(*args, **kwargs)
