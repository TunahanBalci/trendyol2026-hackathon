from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .common import normalize_text


TOKEN_RE = re.compile(r"[\wçğıöşü]+", re.IGNORECASE)
COLORS = {
    "siyah",
    "beyaz",
    "kirmizi",
    "kırmızı",
    "mavi",
    "lacivert",
    "yesil",
    "yeşil",
    "sari",
    "sarı",
    "pembe",
    "mor",
    "bej",
    "gri",
    "kahverengi",
    "turuncu",
    "bordo",
    "haki",
}
MATERIALS = {
    "pamuk",
    "pamuklu",
    "polyester",
    "deri",
    "suni",
    "viskon",
    "keten",
    "yün",
    "yun",
    "akrilik",
    "çelik",
    "celik",
    "plastik",
    "seramik",
    "silikon",
    "ahşap",
    "ahsap",
    "metal",
    "tekstil",
}
SIZE_RE = re.compile(r"\b(?:xs|s|m|l|xl|xxl|xxxl|\d{2,3}(?:\s*x\s*\d{2,3})?|\d{1,2}\s*(?:cm|mm|l|lt|ml|gr|kg))\b")
GENDER_MAP = {
    "kadin": "kadın",
    "kadın": "kadın",
    "bayan": "kadın",
    "erkek": "erkek",
    "unisex": "unisex",
    "kiz": "kız",
    "kız": "kız",
    "cocuk": "çocuk",
    "çocuk": "çocuk",
    "bebek": "bebek",
}


def tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def token_set(text: Any) -> set[str]:
    return set(tokens(text))


def char_ngrams(text: Any, n: int = 3) -> set[str]:
    value = normalize_text(text).replace(" ", "_")
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def query_gender(query: Any) -> str:
    q_tokens = token_set(query)
    for key, value in GENDER_MAP.items():
        if key in q_tokens:
            return value
    return ""


def color_set(text: Any) -> set[str]:
    return token_set(text) & COLORS


def material_set(text: Any) -> set[str]:
    return token_set(text) & MATERIALS


def size_set(text: Any) -> set[str]:
    return set(SIZE_RE.findall(normalize_text(text)))


def category_parts(text: Any) -> list[str]:
    return [part.strip() for part in normalize_text(text).split("/") if part.strip()]


def bm25_score(query_tokens: set[str], field_tokens: list[str], field_stats: dict[str, object] | None) -> float:
    if not query_tokens or not field_tokens or not field_stats:
        return 0.0
    idf: dict[str, float] = field_stats.get("idf", {})  # type: ignore[assignment]
    avg_len = float(field_stats.get("avg_len", 0.0) or 1.0)
    counts = Counter(field_tokens)
    k1 = 1.5
    b = 0.75
    denom_norm = k1 * (1.0 - b + b * len(field_tokens) / max(avg_len, 1.0))
    score = 0.0
    for tok in query_tokens:
        tf = counts.get(tok, 0)
        if tf:
            score += float(idf.get(tok, 0.0)) * (tf * (k1 + 1.0)) / (tf + denom_norm)
    return score


def idf_overlap(query_tokens: set[str], field_tokens: set[str], field_stats: dict[str, object] | None) -> tuple[float, float]:
    if not query_tokens or not field_tokens or not field_stats:
        return 0.0, 0.0
    idf: dict[str, float] = field_stats.get("idf", {})  # type: ignore[assignment]
    query_weight = sum(float(idf.get(tok, 0.0)) for tok in query_tokens)
    overlap_weight = sum(float(idf.get(tok, 0.0)) for tok in query_tokens & field_tokens)
    return overlap_weight, overlap_weight / max(query_weight, 1e-9)


def build_features(df: pd.DataFrame, text_stats: dict[str, dict[str, object]] | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for row in df.itertuples(index=False):
        query = getattr(row, "query", "")
        title = getattr(row, "title", "")
        category = getattr(row, "category", "")
        brand = getattr(row, "brand", "")
        gender = getattr(row, "gender", "")
        age_group = getattr(row, "age_group", "")
        attributes = getattr(row, "attributes", "")
        product_text = getattr(row, "product_text", "")

        q_tokens = token_set(query)
        q_token_list = tokens(query)
        title_token_list = tokens(title)
        product_token_list = tokens(product_text)
        category_token_list = tokens(category)
        attr_token_list = tokens(attributes)
        title_tokens = token_set(title)
        product_tokens = token_set(product_text)
        category_tokens = token_set(category)
        attr_tokens = token_set(attributes)
        q_chars = char_ngrams(query)
        title_chars = char_ngrams(title)
        product_chars = char_ngrams(product_text)

        q_gender = query_gender(query)
        prod_gender = normalize_text(gender)
        q_colors = color_set(query)
        prod_colors = color_set(attributes) | color_set(title)
        q_materials = material_set(query)
        prod_materials = material_set(attributes) | material_set(title)
        q_sizes = size_set(query)
        prod_sizes = size_set(attributes) | size_set(title)
        cat_parts = category_parts(category)

        title_idf_sum, title_idf_ratio = idf_overlap(q_tokens, title_tokens, text_stats.get("title") if text_stats else None)
        product_idf_sum, product_idf_ratio = idf_overlap(
            q_tokens, product_tokens, text_stats.get("product_text") if text_stats else None
        )
        category_idf_sum, category_idf_ratio = idf_overlap(
            q_tokens, category_tokens, text_stats.get("category") if text_stats else None
        )
        attr_idf_sum, attr_idf_ratio = idf_overlap(
            q_tokens, attr_tokens, text_stats.get("attributes") if text_stats else None
        )

        overlap_title = q_tokens & title_tokens
        overlap_product = q_tokens & product_tokens
        rows.append(
            {
                "query_len": len(normalize_text(query)),
                "title_len": len(normalize_text(title)),
                "product_text_len": len(normalize_text(product_text)),
                "query_token_count": len(q_tokens),
                "title_token_count": len(title_tokens),
                "product_token_count": len(product_tokens),
                "title_overlap_count": len(overlap_title),
                "product_overlap_count": len(overlap_product),
                "title_overlap_ratio": len(overlap_title) / max(1, len(q_tokens)),
                "product_overlap_ratio": len(overlap_product) / max(1, len(q_tokens)),
                "title_jaccard": jaccard(q_tokens, title_tokens),
                "product_jaccard": jaccard(q_tokens, product_tokens),
                "category_jaccard": jaccard(q_tokens, category_tokens),
                "attribute_jaccard": jaccard(q_tokens, attr_tokens),
                "category_depth": len(cat_parts),
                "query_top_category_match": int(bool(cat_parts) and cat_parts[0] in q_tokens),
                "query_leaf_category_match": int(bool(cat_parts) and bool(token_set(cat_parts[-1]) & q_tokens)),
                "title_char3_jaccard": jaccard(q_chars, title_chars),
                "product_char3_jaccard": jaccard(q_chars, product_chars),
                "title_bm25": bm25_score(q_tokens, title_token_list, text_stats.get("title") if text_stats else None),
                "product_bm25": bm25_score(
                    q_tokens, product_token_list, text_stats.get("product_text") if text_stats else None
                ),
                "category_bm25": bm25_score(
                    q_tokens, category_token_list, text_stats.get("category") if text_stats else None
                ),
                "attribute_bm25": bm25_score(
                    q_tokens, attr_token_list, text_stats.get("attributes") if text_stats else None
                ),
                "title_idf_overlap_sum": title_idf_sum,
                "title_idf_overlap_ratio": title_idf_ratio,
                "product_idf_overlap_sum": product_idf_sum,
                "product_idf_overlap_ratio": product_idf_ratio,
                "category_idf_overlap_sum": category_idf_sum,
                "category_idf_overlap_ratio": category_idf_ratio,
                "attribute_idf_overlap_sum": attr_idf_sum,
                "attribute_idf_overlap_ratio": attr_idf_ratio,
                "query_in_title": int(normalize_text(query) in normalize_text(title)),
                "all_query_tokens_in_title": int(bool(q_tokens) and q_tokens <= title_tokens),
                "brand_in_query": int(bool(normalize_text(brand)) and normalize_text(brand) in normalize_text(query)),
                "gender_match": int(bool(q_gender) and q_gender == prod_gender),
                "gender_conflict": int(bool(q_gender) and prod_gender not in {"", "unknown", "unisex"} and q_gender != prod_gender),
                "age_baby_query": int("bebek" in q_tokens),
                "age_child_query": int(bool({"çocuk", "cocuk", "kız", "kiz"} & q_tokens)),
                "age_group_baby": int(normalize_text(age_group) == "bebek"),
                "age_group_child": int(normalize_text(age_group) in {"çocuk", "cocuk"}),
                "color_match": int(bool(q_colors) and bool(q_colors & prod_colors)),
                "color_conflict": int(bool(q_colors) and bool(prod_colors) and not bool(q_colors & prod_colors)),
                "material_match": int(bool(q_materials) and bool(q_materials & prod_materials)),
                "material_conflict": int(bool(q_materials) and bool(prod_materials) and not bool(q_materials & prod_materials)),
                "size_match": int(bool(q_sizes) and bool(q_sizes & prod_sizes)),
                "size_conflict": int(bool(q_sizes) and bool(prod_sizes) and not bool(q_sizes & prod_sizes)),
            }
        )

    features = pd.DataFrame(rows)
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    id_cols = [c for c in ["id", "term_id", "item_id", "label", "negative_type"] if c in df.columns]
    return pd.concat([df[id_cols].reset_index(drop=True), features], axis=1)


FEATURE_COLUMNS = [
    "query_len",
    "title_len",
    "product_text_len",
    "query_token_count",
    "title_token_count",
    "product_token_count",
    "title_overlap_count",
    "product_overlap_count",
    "title_overlap_ratio",
    "product_overlap_ratio",
    "title_jaccard",
    "product_jaccard",
    "category_jaccard",
    "attribute_jaccard",
    "category_depth",
    "query_top_category_match",
    "query_leaf_category_match",
    "title_char3_jaccard",
    "product_char3_jaccard",
    "title_bm25",
    "product_bm25",
    "category_bm25",
    "attribute_bm25",
    "title_idf_overlap_sum",
    "title_idf_overlap_ratio",
    "product_idf_overlap_sum",
    "product_idf_overlap_ratio",
    "category_idf_overlap_sum",
    "category_idf_overlap_ratio",
    "attribute_idf_overlap_sum",
    "attribute_idf_overlap_ratio",
    "query_in_title",
    "all_query_tokens_in_title",
    "brand_in_query",
    "gender_match",
    "gender_conflict",
    "age_baby_query",
    "age_child_query",
    "age_group_baby",
    "age_group_child",
    "color_match",
    "color_conflict",
    "material_match",
    "material_conflict",
    "size_match",
    "size_conflict",
]
