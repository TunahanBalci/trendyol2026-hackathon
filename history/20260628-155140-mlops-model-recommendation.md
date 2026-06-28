# Saved Chat: MLOps and Model Recommendation

## User

Currently there is no previous implementation. Which ML method would be best suitable for this competition?
We will build a basic MLOps pipeline for faster and better iterations. Which free MLOps solution would be most suitable? ClearML? MLFlow? something else?
Also, we'll run the code within Docker. The chosen MLOps solution should have high compatability with Docker.

## Assistant

I read `AGENTS.md`, `prompts/competition.md`, and `data/sample.csv`.

**Best ML Method**

Use a **hybrid search relevance classifier with synthetic hard negatives**.

The strongest practical path is:

1. **Generate negatives**
   - Random negatives: same query paired with unrelated products.
   - Hard negatives: same/similar category, overlapping title tokens, same gender/age group, but not known positive.
   - Very hard negatives: BM25 or embedding-nearest products for a query that are not positive pairs.

2. **Train a fast tabular/text baseline**
   - Start with **LightGBM or CatBoost binary classifier**.
   - Features:
     - query-title TF-IDF/BM25 similarity
     - token overlap, character n-gram overlap
     - category path overlap
     - brand match
     - gender/age compatibility
     - parsed attribute matches: color, material, size, model, etc.
     - sentence embedding cosine similarities

3. **Then add a neural reranker**
   - Fine-tune a multilingual/Turkish transformer cross-encoder on:
     - positive pairs
     - generated hard negatives
   - Input format:

```text
query [SEP] title [SEP] category [SEP] brand [SEP] attributes
```

   - Use it either directly or as a feature inside LightGBM.

My recommendation: **do not start with only a transformer**. The competition has positive-only labels and macro-F1, so negative sampling quality and threshold calibration matter as much as model capacity. A LightGBM/CatBoost model with strong hard negatives will iterate faster and expose what matters.

Best target architecture:

```text
candidate pair
  -> text/attribute/category feature builder
  -> LightGBM/CatBoost baseline
  -> optional transformer cross-encoder score
  -> calibrated threshold for macro-F1
  -> submission.csv
```

**MLOps Recommendation**

Use **MLflow + DVC**, not ClearML as the first choice.

MLflow is the best fit for this repo because it is simple, Docker-friendly, free/open-source, and enough for fast competition iteration. DVC is useful alongside MLflow for reproducible pipelines, params, metrics, experiment comparison, and large-data/artifact versioning without bloating Git.

ClearML is also Docker-compatible, but heavier. Its server uses multiple services such as web app, API server, file server, Mongo, Redis, and Elasticsearch; the official repo provides Docker Compose deployment.

So the choice:

```text
Recommended:
MLflow + DVC + Optuna

Use MLflow for:
- experiment tracking
- metrics
- params
- artifacts
- model registry-ish organization

Use DVC for:
- data/pipeline versioning
- reproducible train/eval/submission stages
- params.yaml-driven experiments

Use Optuna for:
- threshold tuning
- LightGBM/CatBoost hyperparameter search
- negative sampling ratio search
```

ClearML would be my second choice only if we want built-in remote job scheduling, queue workers, and a full experiment UI from day one. For a Dockerized Kaggle-style competition pipeline, **MLflow + DVC is leaner and easier to control**.
