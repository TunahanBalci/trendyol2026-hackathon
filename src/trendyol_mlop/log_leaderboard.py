from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import mlflow

from .common import ensure_parent, read_params


def log_leaderboard(params_path: str, public_score: float, note: str = "") -> None:
    params = read_params(params_path)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "public_score": public_score,
        "note": note,
        "submission_path": params["predict"]["output_path"],
    }
    out = ensure_parent("reports/leaderboard.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    mlflow.set_experiment("trendyol-relevance")
    with mlflow.start_run(run_name="leaderboard-score"):
        mlflow.log_metric("public_score", public_score)
        mlflow.log_param("note", note)
        mlflow.log_param("submission_path", params["predict"]["output_path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--public-score", type=float, required=True)
    parser.add_argument("--note", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_leaderboard(args.params, args.public_score, args.note)
