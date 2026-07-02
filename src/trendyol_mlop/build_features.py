from __future__ import annotations

import argparse

import joblib

from .common import read_params, read_table, write_table
from .features import build_features


def run(params_path: str, input_path: str | None = None, output_path: str | None = None) -> None:
    params = read_params(params_path)
    input_path = input_path or params["dataset"]["output_path"]
    output_path = output_path or params["features"]["train_output_path"]
    text_stats = joblib.load(params["features"]["text_stats_path"])
    df = read_table(input_path)
    features = build_features(df, text_stats)
    write_table(features, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.params, args.input, args.output)
