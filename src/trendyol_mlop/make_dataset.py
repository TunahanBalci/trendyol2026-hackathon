from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .common import read_params, read_table, write_table
from .features import COLORS, MATERIALS, color_set, material_set, query_gender, size_set, token_set


def _sample_not_positive(
    rng: np.random.Generator,
    candidates: np.ndarray,
    positive_items: set[str],
    forbidden_items: set[str],
    tries: int = 80,
) -> str | None:
    if len(candidates) == 0:
        return None
    for _ in range(tries):
        item_id = str(rng.choice(candidates))
        if item_id not in forbidden_items and item_id not in positive_items:
            return item_id
    return None


def _append_negative(
    rows: list[dict[str, object]],
    counters: defaultdict[str, int],
    neg_type: str,
    term_id: str,
    item_id: str,
) -> None:
    counters[neg_type] += 1
    rows.append(
        {
            "id": f"NEG_{neg_type}_{counters[neg_type]}",
            "term_id": term_id,
            "item_id": item_id,
            "label": 0,
            "negative_type": neg_type,
        }
    )


def _build_tfidf_candidates(
    positives: pd.DataFrame,
    items: pd.DataFrame,
    terms: pd.DataFrame,
    top_k: int,
    max_features: int,
    batch_size: int,
) -> dict[str, list[str]]:
    term_queries = positives[["term_id"]].drop_duplicates().merge(terms, on="term_id", how="left")
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=max_features)
    item_matrix = vectorizer.fit_transform(items["product_text"].fillna(""))
    query_matrix = vectorizer.transform(term_queries["query"].fillna(""))
    item_ids = items["item_id"].to_numpy()
    candidates: dict[str, list[str]] = {}
    item_matrix_t = item_matrix.T.tocsr()
    term_ids = term_queries["term_id"].to_numpy()
    for start in tqdm(range(0, query_matrix.shape[0], batch_size), desc="tfidf hard-negative batches"):
        end = min(start + batch_size, query_matrix.shape[0])
        sims = (query_matrix[start:end] @ item_matrix_t).tocsr()
        for offset in range(end - start):
            row_start, row_end = sims.indptr[offset], sims.indptr[offset + 1]
            row_indices = sims.indices[row_start:row_end]
            row_scores = sims.data[row_start:row_end]
            if len(row_indices) > top_k:
                chosen = np.argpartition(row_scores, -top_k)[-top_k:]
                chosen = chosen[np.argsort(row_scores[chosen])[::-1]]
            else:
                chosen = np.argsort(row_scores)[::-1]
            candidates[str(term_ids[start + offset])] = [str(item_ids[i]) for i in row_indices[chosen]]
    return candidates


def make_dataset(
    params_path: str,
    max_positives: int | None = None,
    max_items_sample: int | None = None,
    output: str | None = None,
) -> None:
    params = read_params(params_path)
    seed = params["seed"]
    cfg = params["dataset"]
    rng = np.random.default_rng(seed)

    max_positives = cfg["max_positives"] if max_positives is None else max_positives
    max_items_sample = cfg["max_items_sample"] if max_items_sample is None else max_items_sample
    output = output or cfg["output_path"]

    positives = pd.read_csv(params["data"]["train_pairs_path"])
    if max_positives:
        positives = positives.sample(n=min(max_positives, len(positives)), random_state=seed)
    positives = positives.copy()
    positives["label"] = 1
    positives["negative_type"] = "positive"

    terms = read_table(params["data"]["terms_path"])
    items = read_table(params["data"]["catalog_path"])
    positives = positives[positives["item_id"].isin(set(items["item_id"].astype(str)))].copy()
    if positives.empty:
        raise ValueError("No positive training pairs found in prepared catalog.")
    if max_items_sample and max_items_sample < len(items):
        keep_ids = set(positives["item_id"])
        sampled = items.sample(n=max_items_sample, random_state=seed)
        items = pd.concat([sampled, items[items["item_id"].isin(keep_ids)]], ignore_index=True)
        items = items.drop_duplicates("item_id")

    item_ids = items["item_id"].astype(str).to_numpy()
    item_meta = items.set_index("item_id")
    positives_by_term: dict[str, set[str]] = defaultdict(set)
    for term_id, item_id in positives[["term_id", "item_id"]].itertuples(index=False):
        positives_by_term[str(term_id)].add(str(item_id))

    by_top = {
        key: group["item_id"].astype(str).to_numpy()
        for key, group in items.groupby("category_top", dropna=False)
    }
    by_leaf = {
        key: group["item_id"].astype(str).to_numpy()
        for key, group in items.groupby("category_leaf", dropna=False)
    }
    by_gender = {
        key: group["item_id"].astype(str).to_numpy()
        for key, group in items.groupby("gender", dropna=False)
        if key not in {"unknown", "unisex", ""}
    }
    color_to_items: dict[str, np.ndarray] = {}
    item_colors = (
        items[["item_id", "title", "attributes"]]
        .assign(_colors=lambda x: (x["title"].fillna("") + " " + x["attributes"].fillna("")).map(color_set))
    )
    for color in COLORS:
        color_to_items[color] = item_colors[item_colors["_colors"].map(lambda cs: color in cs)]["item_id"].astype(str).to_numpy()
    material_to_items: dict[str, np.ndarray] = {}
    item_materials = (
        items[["item_id", "title", "attributes"]]
        .assign(_materials=lambda x: (x["title"].fillna("") + " " + x["attributes"].fillna("")).map(material_set))
    )
    for material in MATERIALS:
        material_to_items[material] = (
            item_materials[item_materials["_materials"].map(lambda ms: material in ms)]["item_id"].astype(str).to_numpy()
        )
    size_items = (
        items[["item_id", "title", "attributes"]]
        .assign(_sizes=lambda x: (x["title"].fillna("") + " " + x["attributes"].fillna("")).map(size_set))
    )
    items_with_size = size_items[size_items["_sizes"].map(bool)]["item_id"].astype(str).to_numpy()

    query_by_term = terms.set_index("term_id")["query"].to_dict()
    contradiction_cache: dict[str, np.ndarray] = {}

    tfidf_candidates: dict[str, list[str]] = {}
    if cfg["tfidf_negatives_per_positive"] > 0:
        tfidf_candidates = _build_tfidf_candidates(
            positives,
            items,
            terms,
            int(cfg["tfidf_top_k"]),
            int(cfg["tfidf_max_features"]),
            int(cfg["tfidf_batch_size"]),
        )

    neg_rows: list[dict[str, object]] = []
    counters = defaultdict(int)
    used_negatives_by_term: dict[str, set[str]] = defaultdict(set)
    for row in tqdm(positives.itertuples(index=False), total=len(positives), desc="negative sampling"):
        term_id = str(row.term_id)
        pos_item = str(row.item_id)
        pos_items_for_term = positives_by_term[term_id]
        forbidden_items = pos_items_for_term | used_negatives_by_term[term_id]

        for _ in range(int(cfg["random_negatives_per_positive"])):
            neg_item = _sample_not_positive(rng, item_ids, pos_items_for_term, forbidden_items)
            if neg_item:
                _append_negative(neg_rows, counters, "random", term_id, neg_item)
                used_negatives_by_term[term_id].add(neg_item)
                forbidden_items.add(neg_item)

        top = item_meta.loc[pos_item, "category_top"] if pos_item in item_meta.index else ""
        same_top_ids = by_top.get(top, np.array([], dtype=object))
        for _ in range(int(cfg["same_top_negatives_per_positive"])):
            neg_item = _sample_not_positive(rng, same_top_ids, pos_items_for_term, forbidden_items)
            if neg_item:
                _append_negative(neg_rows, counters, "same_top", term_id, neg_item)
                used_negatives_by_term[term_id].add(neg_item)
                forbidden_items.add(neg_item)

        leaf = item_meta.loc[pos_item, "category_leaf"] if pos_item in item_meta.index else ""
        same_leaf_ids = by_leaf.get(leaf, np.array([], dtype=object))
        for _ in range(int(cfg["same_leaf_negatives_per_positive"])):
            neg_item = _sample_not_positive(rng, same_leaf_ids, pos_items_for_term, forbidden_items)
            if neg_item:
                _append_negative(neg_rows, counters, "same_leaf", term_id, neg_item)
                used_negatives_by_term[term_id].add(neg_item)
                forbidden_items.add(neg_item)

        if term_id not in contradiction_cache:
            query = str(query_by_term.get(term_id, ""))
            contradiction_pools: list[np.ndarray] = []
            q_gender = query_gender(query)
            if q_gender:
                contradiction_pools.extend(pool for gender, pool in by_gender.items() if gender != q_gender)
            q_colors = color_set(query)
            if q_colors:
                contradiction_pools.extend(color_to_items[color] for color in (COLORS - q_colors))
            q_materials = material_set(query)
            if q_materials:
                contradiction_pools.extend(material_to_items[material] for material in (MATERIALS - q_materials))
            q_sizes = size_set(query)
            if q_sizes:
                contradiction_pools.append(items_with_size)
            q_tokens = token_set(query)
            if {"telefon", "phone", "iphone", "samsung"} & q_tokens and {"kılıf", "kilif", "kapak"} & q_tokens:
                contradiction_pools.extend(by_leaf.get(leaf, np.array([], dtype=object)) for leaf in ["şarj kabloları", "kablo aksesuarı"])
            pools = [pool for pool in contradiction_pools if len(pool)]
            contradiction_cache[term_id] = np.concatenate(pools) if pools else np.array([], dtype=object)
        contradiction_ids = contradiction_cache[term_id]
        for _ in range(int(cfg["contradiction_negatives_per_positive"])):
            neg_item = _sample_not_positive(rng, contradiction_ids, pos_items_for_term, forbidden_items)
            if neg_item:
                _append_negative(neg_rows, counters, "contradiction", term_id, neg_item)
                used_negatives_by_term[term_id].add(neg_item)
                forbidden_items.add(neg_item)

        hard_items = tfidf_candidates.get(term_id, [])
        added_hard = 0
        for neg_item in hard_items:
            if added_hard >= int(cfg["tfidf_negatives_per_positive"]):
                break
            if neg_item not in forbidden_items and neg_item not in pos_items_for_term:
                added_hard += 1
                _append_negative(neg_rows, counters, "tfidf", term_id, neg_item)
                used_negatives_by_term[term_id].add(neg_item)
                forbidden_items.add(neg_item)

    negatives = pd.DataFrame(neg_rows)
    pairs = pd.concat([positives[["id", "term_id", "item_id", "label", "negative_type"]], negatives], ignore_index=True)
    pairs = pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
    pairs = pairs.dropna(subset=["query", "title"])
    pairs = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    write_table(pairs, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--max-positives", type=int, default=None)
    parser.add_argument("--max-items-sample", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_dataset(args.params, args.max_positives, args.max_items_sample, args.output)
