.PHONY: setup prepare dataset features train predict repro smoke mlflow-up shell clean-artifacts

setup:
	pip install -r requirements.txt

prepare:
	python -m src.trendyol_mlop.prepare_catalog --params params.yaml

dataset:
	python -m src.trendyol_mlop.make_dataset --params params.yaml

validate-dataset:
	python -m src.trendyol_mlop.validate_dataset --params params.yaml

features:
	python -m src.trendyol_mlop.build_features --params params.yaml

train:
	python -m src.trendyol_mlop.train --params params.yaml

predict:
	python -m src.trendyol_mlop.predict --params params.yaml

repro:
	dvc repro

smoke:
	python -m src.trendyol_mlop.prepare_catalog --params params.yaml --max-items 5000 --required-item-ids-csv data/training_pairs.csv
	python -m src.trendyol_mlop.make_dataset --params params.yaml --max-positives 1000 --max-items-sample 5000 --output data/processed/train_pairs_smoke.parquet
	python -m src.trendyol_mlop.validate_dataset --params params.yaml --input data/processed/train_pairs_smoke.parquet --report reports/dataset_report_smoke.json
	python -m src.trendyol_mlop.fit_text_stats --params params.yaml
	python -m src.trendyol_mlop.build_features --input data/processed/train_pairs_smoke.parquet --output data/processed/train_features_smoke.parquet
	python -m src.trendyol_mlop.train --params params.yaml --features data/processed/train_features_smoke.parquet --model-path models/lgbm_smoke.joblib --metrics-path reports/metrics_smoke.json

validate-submission:
	python -m src.trendyol_mlop.validate_submission --params params.yaml

tune:
	python -m src.trendyol_mlop.tune_lgbm --params params.yaml

mlflow-up:
	docker compose up mlflow

shell:
	docker compose run --rm pipeline bash

clean-artifacts:
	rm -rf data/interim data/processed models reports submissions mlruns
