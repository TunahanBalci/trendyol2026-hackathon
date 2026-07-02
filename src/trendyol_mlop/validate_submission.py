from __future__ import annotations

import argparse

import pandas as pd

from .common import read_params, write_json


def validate_submission(params_path: str, submission_path: str | None = None) -> None:
    params = read_params(params_path)
    submission_path = submission_path or params["predict"]["output_path"]
    sample_path = params["data"]["sample_submission_path"]

    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path, usecols=["id"])
    errors: list[str] = []

    if list(sub.columns) != ["id", "prediction"]:
        errors.append("columns must be exactly: id,prediction")
    if len(sub) != len(sample):
        errors.append(f"row count mismatch: submission={len(sub)} sample={len(sample)}")
    if sub["id"].duplicated().any():
        errors.append("duplicate ids found")
    if not set(sub["prediction"].dropna().unique()).issubset({0, 1}):
        errors.append("prediction must contain only 0/1")
    if set(sub["id"]) != set(sample["id"]):
        errors.append("id set mismatch vs sample_submission.csv")

    report = {
        "path": submission_path,
        "rows": int(len(sub)),
        "positive_rate": float(sub["prediction"].mean()) if len(sub) else 0.0,
        "prediction_counts": {str(k): int(v) for k, v in sub["prediction"].value_counts().to_dict().items()},
        "errors": errors,
        "valid": not errors,
    }
    write_json(report, "reports/submission_report.json")
    if errors:
        raise ValueError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--submission", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_submission(args.params, args.submission)
