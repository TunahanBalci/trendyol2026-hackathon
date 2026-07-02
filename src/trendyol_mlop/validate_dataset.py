from __future__ import annotations

import argparse

import pandas as pd

from .common import read_params, read_table, write_json


REQUIRED_COLUMNS = {
    "id",
    "term_id",
    "item_id",
    "query",
    "title",
    "category",
    "brand",
    "gender",
    "age_group",
    "attributes",
    "label",
    "negative_type",
}


def validate_dataset(params_path: str, input_path: str | None = None, report_path: str | None = None) -> None:
    params = read_params(params_path)
    cfg = params["dataset"]
    input_path = input_path or cfg["output_path"]
    report_path = report_path or cfg["report_path"]

    df = read_table(input_path)
    errors: list[str] = []
    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
    if "label" in df and not set(df["label"].dropna().unique()).issubset({0, 1}):
        errors.append("label must contain only 0/1")
    if "id" in df and df["id"].duplicated().any():
        errors.append("duplicate ids found")
    if {"term_id", "item_id", "label"}.issubset(df.columns):
        pair_dupes = int(df.duplicated(["term_id", "item_id"]).sum())
        if pair_dupes:
            errors.append(f"duplicate term/item pairs: {pair_dupes}")
        positives = df[df["label"] == 1][["term_id", "item_id"]].drop_duplicates()
        negatives = df[df["label"] == 0][["term_id", "item_id"]].drop_duplicates()
        overlap = positives.merge(negatives, on=["term_id", "item_id"])
        if len(overlap):
            errors.append(f"positive/negative pair overlap: {len(overlap)}")

    for col in ["query", "title", "category", "brand", "gender", "age_group", "attributes"]:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            empty = int((df[col].fillna("").astype(str).str.len() == 0).sum())
            if col in {"query", "title"} and (missing or empty):
                errors.append(f"{col} has missing/empty values: missing={missing}, empty={empty}")

    positives_n = int((df["label"] == 1).sum()) if "label" in df else 0
    negatives_n = int((df["label"] == 0).sum()) if "label" in df else 0
    ratio = negatives_n / max(positives_n, 1)
    if positives_n < int(cfg["min_positive_rows"]):
        errors.append(f"too few positives: {positives_n}")
    if ratio < float(cfg["min_negative_per_positive"]):
        errors.append(f"negative/positive ratio too low: {ratio:.3f}")

    report = {
        "path": input_path,
        "rows": int(len(df)),
        "positive_rows": positives_n,
        "negative_rows": negatives_n,
        "negative_per_positive": float(ratio),
        "negative_type_counts": {
            str(k): int(v) for k, v in df.get("negative_type", pd.Series(dtype=str)).value_counts().to_dict().items()
        },
        "duplicate_pairs": int(df.duplicated(["term_id", "item_id"]).sum())
        if {"term_id", "item_id"}.issubset(df.columns)
        else None,
        "missing_counts": {
            col: int(df[col].isna().sum()) for col in sorted(REQUIRED_COLUMNS & set(df.columns))
        },
        "errors": errors,
        "valid": not errors,
    }
    write_json(report, report_path)
    if errors:
        raise ValueError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(args.params, args.input, args.report)
