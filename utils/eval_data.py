"""Data loading for the evaluation pipeline."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd


DEFAULT_DATASET = "JunkaiZ/MatSciBench"
DEFAULT_SPLIT = "test"
LOCAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "MatSciBench" / "MatSciBench.parquet"


def _is_repo_path(path_entry: str, repo_root: Path) -> bool:
    path = Path(path_entry or os.getcwd())
    try:
        return path.resolve() == repo_root
    except OSError:
        return False


def _import_hf_datasets():
    existing = sys.modules.get("datasets")
    if existing is not None and hasattr(existing, "load_dataset"):
        return existing

    removed_module = None
    if existing is not None:
        removed_module = sys.modules.pop("datasets", None)

    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    sys.path = [entry for entry in sys.path if not _is_repo_path(entry, repo_root)]
    try:
        try:
            module = importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            if removed_module is not None:
                sys.modules["datasets"] = removed_module
            raise ImportError(
                "Could not import Hugging Face datasets.load_dataset. "
                "Install the `datasets` package to load the remote benchmark."
            ) from exc
    finally:
        sys.path = original_path

    if not hasattr(module, "load_dataset"):
        if removed_module is not None:
            sys.modules["datasets"] = removed_module
        raise ImportError(
            "Could not import Hugging Face datasets.load_dataset. "
            "Install the `datasets` package and ensure it is not shadowed by a local module."
        )
    return module


def _normalize_image_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    try:
        if value is None or pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return []


def _load_local_parquet(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    rows: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        normalized = dict(row)
        normalized["image"] = _normalize_image_list(normalized.get("image"))
        rows.append(normalized)
    return rows


def _resolve_local_dataset(dataset: str) -> Path | None:
    candidate = Path(dataset)
    if candidate.is_file() and candidate.suffix == ".parquet":
        return candidate
    if candidate.is_dir():
        parquet_path = candidate / "MatSciBench.parquet"
        if parquet_path.is_file():
            return parquet_path
    return None


def load_eval_data(dataset: str = DEFAULT_DATASET, split: str = DEFAULT_SPLIT) -> List[Dict[str, Any]]:
    local_dataset = _resolve_local_dataset(dataset)
    if local_dataset is not None:
        return _load_local_parquet(local_dataset)

    try:
        hf_dataset = _import_hf_datasets().load_dataset(dataset, split=split)
    except Exception:
        if dataset == DEFAULT_DATASET and LOCAL_DATASET.is_file():
            print(f"Falling back to local benchmark parquet: {LOCAL_DATASET}")
            return _load_local_parquet(LOCAL_DATASET)
        raise

    rows: List[Dict[str, Any]] = []
    for row in hf_dataset:
        normalized = dict(row)
        normalized["image"] = list(normalized.get("image") or [])
        rows.append(normalized)
    return rows
