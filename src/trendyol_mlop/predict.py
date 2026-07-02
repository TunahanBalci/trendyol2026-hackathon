from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_parent, read_params, read_table, write_json
from .features import build_features
from .validate_submission import validate_submission


def predict(params_path: str, output_path: str | None = None) -> None:
    params = read_params(params_path)
    output_path = output_path or params["predict"]["output_path"]
    chunksize = int(params["predict"]["chunksize"])

    terms = read_table(params["data"]["terms_path"])
    items = read_table(params["data"]["catalog_path"])
    model_bundle = joblib.load(params["train"]["model_path"])
    text_stats = joblib.load(params["features"]["text_stats_path"])
    model = model_bundle["model"]
    threshold = float(model_bundle["threshold"])
    feature_cols = model_bundle["feature_columns"]

    out = ensure_parent(output_path)
    first = True
    probas: list[np.ndarray] = []
    missing_terms = 0
    missing_items = 0
    total = sum(1 for _ in open(params["data"]["submission_pairs_path"], "r", encoding="utf-8")) - 1
    reader = pd.read_csv(params["data"]["submission_pairs_path"], chunksize=chunksize)
    for chunk in tqdm(reader, total=(total // chunksize) + 1, desc="predict chunks"):
        merged = chunk.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
        missing_terms += int(merged["query"].isna().sum())
        missing_items += int(merged["title"].isna().sum())
        feats = build_features(merged, text_stats)
        proba = model.predict_proba(feats[feature_cols])[:, 1]
        probas.append(proba)
        sub = pd.DataFrame({"id": chunk["id"], "prediction": (proba >= threshold).astype(int)})
        sub.to_csv(out, mode="w" if first else "a", index=False, header=first)
        first = False
    all_proba = np.concatenate(probas) if probas else np.array([])
    report = {
        "rows": int(len(all_proba)),
        "threshold": threshold,
        "positive_rate": float((all_proba >= threshold).mean()) if len(all_proba) else 0.0,
        "missing_terms": missing_terms,
        "missing_items": missing_items,
        "score_quantiles": {
            str(q): float(np.quantile(all_proba, q)) for q in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        }
        if len(all_proba)
        else {},
    }
    write_json(report, params["predict"]["score_report_path"])
    validate_submission(params_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.params, args.output)
