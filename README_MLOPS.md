# MLOps Pipeline

Purpose: fast, reproducible search-relevance experiments for positive-only Trendyol data.

## Lifecycle

```text
data/items.csv + terms.csv
  -> prepare_catalog
  -> make_dataset
  -> fit_text_stats
  -> build_features
  -> train
  -> predict
```

## Run Locally

```bash
pip install -r requirements.txt
make smoke
make repro
```

## Run With Docker

```bash
docker compose up -d mlflow
docker compose run --rm pipeline dvc repro
```

MLflow UI:

```text
http://localhost:5000
```

## What Gets Tracked

- `params.yaml`: data generation, model, threshold settings
- `dvc.yaml`: reproducible stages and dependencies
- MLflow: params, metrics, model artifact, feature columns
- `reports/metrics.json`: local validation macro-F1 and class F1
- `submissions/submission.csv`: generated competition file
- `reports/submission_report.json`: ID/row/prediction-rate checks

## Fast Iteration Loop

1. Change negative sampling in `params.yaml`.
2. Run `dvc repro`.
3. Compare MLflow runs.
4. Submit best `submissions/submission.csv`.
5. Record public score next to run metadata.

## First Useful Knobs

- `dataset.random_negatives_per_positive`
- `dataset.same_top_negatives_per_positive`
- `dataset.tfidf_negatives_per_positive`
- `dataset.same_leaf_negatives_per_positive`
- `dataset.contradiction_negatives_per_positive`
- `dataset.max_items_sample`
- `features.max_vocab`
- `train.model.*`
- selected threshold in `reports/metrics.json`
