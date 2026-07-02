from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from .common import ensure_parent, read_params, read_table, write_json
from .features import FEATURE_COLUMNS


def _make_model(params: dict) -> object:
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=params["seed"],
            n_jobs=-1,
            verbose=-1,
            **params["train"]["model"],
        )
    except Exception:
        return HistGradientBoostingClassifier(
            learning_rate=params["train"]["model"]["learning_rate"],
            max_iter=params["train"]["model"]["n_estimators"],
            random_state=params["seed"],
        )


def _best_threshold(y_true: np.ndarray, proba: np.ndarray, grid_size: int) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.0, 1.0, grid_size):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def _log_mlflow(metrics: dict, params: dict, model_path: str, feature_cols: list[str]) -> None:
    try:
        import mlflow

        experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "trendyol-relevance")
        mlflow.set_experiment(experiment)
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "seed": params["seed"],
                    "random_negatives_per_positive": params["dataset"]["random_negatives_per_positive"],
                    "same_top_negatives_per_positive": params["dataset"]["same_top_negatives_per_positive"],
                    "same_leaf_negatives_per_positive": params["dataset"]["same_leaf_negatives_per_positive"],
                    "contradiction_negatives_per_positive": params["dataset"]["contradiction_negatives_per_positive"],
                    "tfidf_negatives_per_positive": params["dataset"]["tfidf_negatives_per_positive"],
                    "max_items_sample": params["dataset"]["max_items_sample"],
                    "primary_split": params["train"]["primary_split"],
                    **{f"model_{k}": v for k, v in params["train"]["model"].items()},
                }
            )
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})
            mlflow.log_text("\n".join(feature_cols), "feature_columns.txt")
            mlflow.log_artifact(model_path)
    except Exception as exc:
            print(f"MLflow logging skipped: {exc}")


def _split_indices(df, y: np.ndarray, split_name: str, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_placeholder = np.zeros(len(df))
    if split_name == "random":
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(splitter.split(x_placeholder, y))
    if split_name == "item":
        groups = df["item_id"].astype(str).to_numpy()
    elif split_name == "term":
        groups = df["term_id"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown validation split: {split_name}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(x_placeholder, y, groups))


def _evaluate_split(
    params: dict,
    df,
    feature_cols: list[str],
    y: np.ndarray,
    split_name: str,
) -> tuple[dict[str, float | int], float]:
    cfg = params["train"]
    x = df[feature_cols]
    train_idx, valid_idx = _split_indices(df, y, split_name, cfg["test_size"], params["seed"])
    model = _make_model(params)
    model.fit(x.iloc[train_idx], y[train_idx])

    proba = model.predict_proba(x.iloc[valid_idx])[:, 1]
    threshold, macro_f1 = _best_threshold(y[valid_idx], proba, int(cfg["threshold_grid_size"]))
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y[valid_idx], pred, labels=[0, 1], zero_division=0
    )
    metrics: dict[str, float | int] = {
        f"{split_name}_macro_f1": float(macro_f1),
        f"{split_name}_threshold": float(threshold),
        f"{split_name}_f1_irrelevant": float(f1[0]),
        f"{split_name}_f1_relevant": float(f1[1]),
        f"{split_name}_precision_irrelevant": float(precision[0]),
        f"{split_name}_precision_relevant": float(precision[1]),
        f"{split_name}_recall_irrelevant": float(recall[0]),
        f"{split_name}_recall_relevant": float(recall[1]),
        f"{split_name}_support_irrelevant": int(support[0]),
        f"{split_name}_support_relevant": int(support[1]),
        f"{split_name}_train_rows": int(len(train_idx)),
        f"{split_name}_valid_rows": int(len(valid_idx)),
        f"{split_name}_positive_rate_pred": float(pred.mean()),
        f"{split_name}_positive_rate_true": float(y[valid_idx].mean()),
    }
    valid_df = df.iloc[valid_idx].copy()
    valid_df["_y"] = y[valid_idx]
    valid_df["_pred"] = pred
    for neg_type in sorted(t for t in valid_df["negative_type"].dropna().unique() if t != "positive"):
        neg_mask = valid_df["negative_type"] == neg_type
        subset_mask = neg_mask | (valid_df["negative_type"] == "positive")
        neg_pred = valid_df.loc[neg_mask, "_pred"]
        metrics[f"{split_name}_{neg_type}_count"] = int(neg_mask.sum())
        metrics[f"{split_name}_{neg_type}_false_positive_rate"] = float(neg_pred.mean()) if len(neg_pred) else 0.0
        if subset_mask.sum() and valid_df.loc[subset_mask, "_y"].nunique() == 2:
            metrics[f"{split_name}_{neg_type}_macro_f1"] = float(
                f1_score(
                    valid_df.loc[subset_mask, "_y"].to_numpy(),
                    valid_df.loc[subset_mask, "_pred"].to_numpy(),
                    average="macro",
                    zero_division=0,
                )
            )
    return metrics, threshold


def _write_feature_importance(model: object, feature_cols: list[str], path: str) -> None:
    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
    else:
        values = np.zeros(len(feature_cols))
    out = pd.DataFrame({"feature": feature_cols, "importance": values})
    out = out.sort_values("importance", ascending=False)
    ensure_parent(path)
    out.to_csv(path, index=False)


def train(params_path: str, features_path: str | None, model_path: str | None, metrics_path: str | None) -> None:
    params = read_params(params_path)
    cfg = params["train"]
    features_path = features_path or params["features"]["train_output_path"]
    model_path = model_path or cfg["model_path"]
    metrics_path = metrics_path or cfg["metrics_path"]

    df = read_table(features_path)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[feature_cols]
    y = df["label"].astype(int).to_numpy()

    metrics: dict[str, float | int] = {
        "rows": int(len(df)),
        "features": int(len(feature_cols)),
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
    }
    thresholds: dict[str, float] = {}
    for split_name in cfg["validation_splits"]:
        split_metrics, split_threshold = _evaluate_split(params, df, feature_cols, y, split_name)
        metrics.update(split_metrics)
        thresholds[split_name] = split_threshold

    primary_split = cfg["primary_split"]
    threshold = thresholds[primary_split]
    metrics["macro_f1"] = float(metrics[f"{primary_split}_macro_f1"])
    metrics["threshold"] = float(threshold)
    metrics["primary_split"] = primary_split  # type: ignore[assignment]

    model = _make_model(params)
    if cfg.get("train_final_on_all", True):
        model.fit(x, y)
        metrics["final_train_rows"] = int(len(df))
    else:
        train_idx, _ = _split_indices(df, y, primary_split, cfg["test_size"], params["seed"])
        model.fit(x.iloc[train_idx], y[train_idx])
        metrics["final_train_rows"] = int(len(train_idx))

    ensure_parent(model_path)
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": feature_cols}, model_path)
    _write_feature_importance(model, feature_cols, cfg["feature_importance_path"])
    write_json(metrics, metrics_path)
    _log_mlflow(metrics, params, model_path, feature_cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.params, args.features, args.model_path, args.metrics_path)
