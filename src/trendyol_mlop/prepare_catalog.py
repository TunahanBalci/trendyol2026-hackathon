from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import leaf_category, read_params, top_category, write_table


ITEM_COLUMNS = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]


def prepare_catalog(
    params_path: str,
    max_items: int | None = None,
    chunksize: int = 100_000,
    required_item_ids_csv: str | None = None,
) -> None:
    params = read_params(params_path)
    raw_dir = Path(params["data"]["raw_dir"])
    catalog_path = params["data"]["catalog_path"]
    terms_path = params["data"]["terms_path"]

    required_ids: set[str] = set()
    if required_item_ids_csv:
        required = pd.read_csv(required_item_ids_csv, usecols=["item_id"])
        required_ids = set(required["item_id"].astype(str))

    frames: list[pd.DataFrame] = []
    seen = 0
    reader = pd.read_csv(raw_dir / "items.csv", usecols=ITEM_COLUMNS, chunksize=chunksize)
    for chunk in tqdm(reader, desc="items chunks"):
        if max_items is not None:
            sampled = chunk.head(max(0, max_items - seen))
            required_chunk = chunk[chunk["item_id"].astype(str).isin(required_ids)]
            chunk = pd.concat([sampled, required_chunk], ignore_index=True).drop_duplicates("item_id")
        chunk = chunk.copy()
        chunk["category_top"] = chunk["category"].map(top_category)
        chunk["category_leaf"] = chunk["category"].map(leaf_category)
        chunk["product_text"] = (
            chunk["title"].fillna("")
            + " "
            + chunk["category"].fillna("")
            + " "
            + chunk["brand"].fillna("")
            + " "
            + chunk["gender"].fillna("")
            + " "
            + chunk["age_group"].fillna("")
            + " "
            + chunk["attributes"].fillna("")
        )
        frames.append(chunk)
        seen += len(chunk)
        if max_items is not None and not required_ids and seen >= max_items:
            break

    catalog = pd.concat(frames, ignore_index=True)
    write_table(catalog, catalog_path)

    terms = pd.read_csv(raw_dir / "terms.csv")
    write_table(terms, terms_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--required-item-ids-csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_catalog(args.params, args.max_items, args.chunksize, args.required_item_ids_csv)
