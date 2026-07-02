from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = ensure_parent(path)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).lower().strip()


def top_category(category: Any) -> str:
    text = normalize_text(category)
    return text.split("/")[0].strip() if text else ""


def leaf_category(category: Any) -> str:
    text = normalize_text(category)
    return text.split("/")[-1].strip() if text else ""
