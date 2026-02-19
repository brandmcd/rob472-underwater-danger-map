from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config file: {p}")
    with p.open("r") as f:
        return yaml.safe_load(f)


def _expand(s: str) -> str:
    # supports ${USER} etc.
    return os.path.expandvars(s)


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    images_dir: Path
    labels_dir: Optional[Path]
    has_labels: bool


def resolve_dataset_paths(
    *,
    profile: str,
    dataset: str,
    profiles_yaml: str | Path = "configs/profiles.yaml",
    datasets_yaml: str | Path = "configs/datasets.yaml",
) -> DatasetPaths:
    profiles = _load_yaml(profiles_yaml)
    datasets = _load_yaml(datasets_yaml)

    # profiles.yaml format:
    # profiles:
    #   local:
    #     data_root: ...
    #     outputs_root: ...
    prof = profiles.get("profiles", {}).get(profile)
    if prof is None:
        raise KeyError(f"Profile '{profile}' not found in {profiles_yaml}")

    data_root = Path(_expand(str(prof["data_root"]))).resolve()

    # datasets.yaml format:
    # datasets:
    #   suim:
    #     images_rel: ...
    #     labels_rel: ...
    #     has_labels: true/false
    ds = datasets.get("datasets", {}).get(dataset)
    if ds is None:
        raise KeyError(f"Dataset '{dataset}' not found in {datasets_yaml}")

    images_rel = ds.get("images_rel")
    labels_rel = ds.get("labels_rel")
    has_labels = bool(ds.get("has_labels", False))

    if not images_rel:
        raise ValueError(f"{datasets_yaml}: datasets.{dataset}.images_rel is missing")

    images_dir = (data_root / _expand(images_rel)).resolve()
    labels_dir = None
    if labels_rel:
        labels_dir = (data_root / _expand(labels_rel)).resolve()

    return DatasetPaths(
        name=dataset,
        images_dir=images_dir,
        labels_dir=labels_dir,
        has_labels=has_labels,
    )