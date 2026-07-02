# Graph Report - .  (2026-07-02)

## Corpus Check
- Corpus is ~5,929 words - fits in a single context window. You may not need a graph.

## Summary
- 124 nodes · 280 edges · 14 communities (7 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `033e1d5b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_read_params|read_params]]
- [[_COMMUNITY_features.py|features.py]]
- [[_COMMUNITY_Training and Validation|Training and Validation]]
- [[_COMMUNITY_train.py|train.py]]
- [[_COMMUNITY_Synthetic Negative Sampling|Synthetic Negative Sampling]]
- [[_COMMUNITY_MLOps Pipeline|MLOps Pipeline]]
- [[_COMMUNITY_parse_args|parse_args]]
- [[_COMMUNITY_parse_args|parse_args]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_parse_args|parse_args]]
- [[_COMMUNITY_parse_args|parse_args]]
- [[_COMMUNITY_parse_args|parse_args]]
- [[_COMMUNITY_parse_args|parse_args]]

## God Nodes (most connected - your core abstractions)
1. `read_params()` - 23 edges
2. `read_table()` - 18 edges
3. `build_features()` - 18 edges
4. `write_json()` - 14 edges
5. `make_dataset()` - 14 edges
6. `ensure_parent()` - 13 edges
7. `write_table()` - 10 edges
8. `normalize_text()` - 10 edges
9. `train()` - 10 edges
10. `token_set()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Public/private leaderboard risk` --tradeoff_after_validation--> `Final model retrains on all synthetic pairs`  [INFERRED]
  AGENTS.md → params.yaml
- `Target score >=0.91 public with private stability` --drives--> `MLflow + DVC MLOps pipeline`  [EXTRACTED]
  AGENTS.md → README_MLOPS.md
- `Positive-only incomplete training pairs` --requires--> `Tiered synthetic negative sampling strategy`  [EXTRACTED]
  AGENTS.md → history/20260628-161116-data-creation-plan.md
- `items.csv large-file sampling rule` --constrains--> `DVC stage prepare_catalog`  [EXTRACTED]
  AGENTS.md → dvc.yaml
- `Docker Compose MLflow service` --hosts_tracking_for--> `MLflow + DVC MLOps pipeline`  [EXTRACTED]
  docker-compose.yml → README_MLOPS.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Dockerized MLflow/DVC experiment stack** — mlops_pipeline, docker_mlflow_service, docker_pipeline_service, mlflow_tracking_uri, requirements_ml_stack [EXTRACTED 1.00]
- **Synthetic negative generation family** — negative_sampling_strategy, negative_random, negative_category, negative_hard_tfidf, negative_contradiction, stage_make_dataset [EXTRACTED 1.00]
- **Search relevance DVC lifecycle** — stage_prepare_catalog, stage_make_dataset, stage_validate_dataset, stage_fit_text_stats, stage_build_features, stage_train, stage_predict [EXTRACTED 1.00]

## Communities (14 total, 7 thin omitted)

### Community 0 - "read_params"
Cohesion: 0.24
Nodes (19): Path, run(), ensure_parent(), leaf_category(), Any, DataFrame, read_params(), read_table() (+11 more)

### Community 1 - "features.py"
Cohesion: 0.21
Nodes (24): defaultdict, normalize_text(), bm25_score(), build_features(), category_parts(), char_ngrams(), color_set(), idf_overlap() (+16 more)

### Community 2 - "Training and Validation"
Cohesion: 0.13
Nodes (16): models/lgbm_relevance.joblib, reports/metrics.json, submissions/submission.csv, reports/submission_report.json, Public/private leaderboard risk, Macro-F1 threshold calibration, Hybrid search relevance classifier with LightGBM, Optional transformer cross-encoder reranker (+8 more)

### Community 3 - "train.py"
Cohesion: 0.27
Nodes (13): _best_threshold(), _evaluate_split(), _log_mlflow(), _make_model(), parse_args(), Namespace, ndarray, _split_indices() (+5 more)

### Community 4 - "Synthetic Negative Sampling"
Cohesion: 0.21
Nodes (14): reports/dataset_report.json, data/interim/text_stats.joblib, data/processed/train_features.parquet, data/processed/train_pairs_synth.parquet, Positive-only incomplete training pairs, Category-aware negatives, Contradiction negatives, TF-IDF hard negatives (+6 more)

### Community 5 - "MLOps Pipeline"
Cohesion: 0.18
Nodes (11): data/interim/items_min.parquet, data/interim/terms.parquet, items.csv large-file sampling rule, Target score >=0.91 public with private stability, Docker Compose MLflow service, Docker Compose pipeline service, MLFLOW_TRACKING_URI=http://mlflow:5000, MLflow + DVC MLOps pipeline (+3 more)

## Knowledge Gaps
- **18 isolated node(s):** `Target score >=0.91 public with private stability`, `Positive-only incomplete training pairs`, `items.csv large-file sampling rule`, `data/interim/items_min.parquet`, `data/interim/terms.parquet` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `read_params()` connect `read_params` to `features.py`, `train.py`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `read_table()` connect `read_params` to `features.py`, `train.py`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `DVC stage train` connect `Training and Validation` to `Synthetic Negative Sampling`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **What connects `Trendyol relevance MLOps pipeline.`, `Target score >=0.91 public with private stability`, `Positive-only incomplete training pairs` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Training and Validation` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._