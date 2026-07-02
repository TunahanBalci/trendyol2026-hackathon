from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mlflow
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

from .common import read_params, read_table, write_json
from .features import FEATURE_COLUMNS
from .train import _best_threshold, _split_indices


def tune(params_path: str) -> None:
    params = read_params(params_path)
    cfg = params["train"]
    tune_cfg = params["tune"]
    df = read_table(params["features"]["train_output_path"])
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[feature_cols]
    y = df["label"].astype(int).to_numpy()
    train_idx, valid_idx = _split_indices(df, y, cfg["primary_split"], cfg["test_size"], params["seed"])

    def objective(trial: optuna.Trial) -> float:
        model = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=params["seed"],
            n_jobs=-1,
            verbose=-1,
            n_estimators=trial.suggest_int("n_estimators", 400, 1600),
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
            reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(x.iloc[valid_idx])[:, 1]
        threshold, score = _best_threshold(y[valid_idx], proba, int(cfg["threshold_grid_size"]))
        trial.set_user_attr("threshold", threshold)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(tune_cfg["trials"]))
    best = {
        "best_score": float(study.best_value),
        "best_threshold": float(study.best_trial.user_attrs["threshold"]),
        "best_params": study.best_params,
        "split": cfg["primary_split"],
        "trials": int(tune_cfg["trials"]),
    }
    write_json(best, tune_cfg["output_path"])

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "trendyol-relevance"))
    with mlflow.start_run(run_name="optuna-lgbm"):
        mlflow.log_metric("best_score", best["best_score"])
        mlflow.log_metric("best_threshold", best["best_threshold"])
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_artifact(tune_cfg["output_path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune(args.params)
