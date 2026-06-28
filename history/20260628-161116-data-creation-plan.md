# Saved Chat: Data Creation Plan

## User

How should we approach this creating data plan? Can a python script do this? will it be enough? what will be the data quality?

## Assistant

Yes, a Python script can absolutely generate the first useful training dataset. It will be enough for a strong baseline, but not enough to maximize leaderboard by itself unless we iterate on negative quality and validation.

The core plan should be:

```text
positive pairs from training_pairs.csv
+ generated negative pairs
+ joined term/product text
+ feature extraction
+ validation split by term_id/item_id
```

**Data Creation Strategy**

We should create negatives in tiers, not one random pool.

1. **Easy random negatives**
   - For each positive `(term_id, item_id)`, sample unrelated items.
   - Avoid the original positive item.
   - These teach the model obvious irrelevance.
   - Quality: high precision negatives, but too easy.

2. **Category-aware negatives**
   - Pair the query with items from different categories.
   - Also pair with items from nearby but wrong categories.
   - Example: query for "kadin siyah elbise" gets unrelated shoes, bags, men's shirts.
   - Quality: good, still mostly safe.

3. **Hard negatives**
   - Same top category or same leaf category, but different product intent.
   - Example: query "siyah elbise" with black blouse, black skirt, red dress.
   - These are essential for macro-F1.
   - Quality: lower purity, but much more valuable.

4. **BM25/TF-IDF hard negatives**
   - Build a lightweight search index over title/category/attributes.
   - For each query, retrieve top-k products.
   - Exclude known positive pairs.
   - Sample from high-similarity products as negatives.
   - Quality: mixed. Some false negatives will exist, but this distribution is closest to test.

5. **Contradiction negatives**
   - Use attribute conflicts:
     - query says `kadin`, product is `erkek`
     - query says `siyah`, product color is `beyaz`
     - query says `telefon kilifi`, product is charger/cable
   - Quality: high if rules are conservative.

**Will A Python Script Be Enough?**

For the first phase, yes.

A single script like this is enough:

```text
scripts/make_dataset.py
  --neg-ratio 5
  --hard-neg-ratio 0.6
  --seed 42
  --output data/processed/train.parquet
```

It should output:

```text
id
term_id
item_id
query
title
category
brand
gender
age_group
attributes
label
negative_type
```

Then another script can build features:

```text
scripts/build_features.py
```

And training:

```text
scripts/train_lgbm.py
```

Eventually, we should move this into DVC stages, but plain Python scripts are the right start.

**Expected Data Quality**

Initial quality will be acceptable but noisy.

Rough expectation:

```text
easy random negatives:      95-99% truly negative
category negatives:         90-97% truly negative
hard negatives:             75-95% truly negative
BM25 hard negatives:         60-90% truly negative
```

The risk is **false negatives**: products we label `0` that are actually relevant. That is unavoidable because the official training data only contains positives and is incomplete.

To control this, we should:

- keep `negative_type` as a column
- train/evaluate by negative type
- avoid using only BM25 hard negatives
- use multiple negative ratios
- tune the prediction threshold for macro-F1
- create validation sets that mimic test difficulty

My recommendation for the first dataset:

```text
For each positive:
- 2 random negatives
- 2 category-aware negatives
- 2 BM25/TF-IDF hard negatives

Total ratio: 1 positive : 6 negatives
```

Then train LightGBM/CatBoost and compare against ratios like `1:3`, `1:6`, `1:10`.

The key is not perfect synthetic labels. The key is producing a negative distribution close enough to the hidden test set, then calibrating the decision threshold well.
