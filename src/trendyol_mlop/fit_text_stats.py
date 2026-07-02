from __future__ import annotations

import argparse
from collections import Counter

import joblib
import numpy as np
from tqdm import tqdm

from .common import ensure_parent, read_params, read_table
from .features import tokens


FIELDS = ["title", "category", "attributes", "product_text"]


def fit_text_stats(params_path: str) -> None:
    params = read_params(params_path)
    cfg = params["features"]
    items = read_table(params["data"]["catalog_path"])
    max_vocab = int(cfg["max_vocab"])
    min_df = int(cfg["min_df"])

    stats: dict[str, dict[str, object]] = {}
    for field in FIELDS:
        doc_freq: Counter[str] = Counter()
        lengths: list[int] = []
        for text in tqdm(items[field].fillna(""), desc=f"idf {field}"):
            toks = tokens(text)
            lengths.append(len(toks))
            doc_freq.update(set(toks))

        kept = [(tok, df) for tok, df in doc_freq.items() if df >= min_df]
        kept.sort(key=lambda x: x[1], reverse=True)
        kept = kept[:max_vocab]
        n_docs = len(items)
        idf = {tok: float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))) for tok, df in kept}
        stats[field] = {
            "idf": idf,
            "avg_len": float(np.mean(lengths) if lengths else 0.0),
            "n_docs": int(n_docs),
        }

    out = ensure_parent(cfg["text_stats_path"])
    joblib.dump(stats, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fit_text_stats(args.params)
