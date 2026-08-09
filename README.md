# Gaming Backlog Predictor 🎮

A personal project to answer one question: **which game from my backlog should I play next?**

The original idea was simple — build a small backlog prediction system and have fun with ML along the way. It stayed simple in scope, but it became a playground for genuinely cool techniques: custom sklearn transformers, target encoding with shrinkage, LOO-CV, SHAP explainability, and even a zero-shot tabular foundation model.

---

## How It Works

I keep my backlog in a Google Sheet, where each game has a genre, franchise, developer, status, and — for the games I've finished — a personal rating (`Nota`). The pipeline trains a regression model on the finished games and ranks the unplayed ones by predicted rating.

```
Google Sheets (or cached CSV)
     │
     ▼
load_sheet()          ← live via gspread OAuth, or local=True for the cached data/backlog.csv
     │
     ▼
prepare_data()        ← splits into finished (train) / backlog (infer)
     │
     ▼
BacklogEncoder        ← multi-hot genres + James-Stein franchise + M-Estimate developer + numeric scores
     │
     ▼
SimpleImputer → StandardScaler
     │
     ▼
RidgeCV               ← LOO-CV evaluation → fit on full training set → save data/model.pkl
     │
     ▼
predict_backlog()     ← scores every backlog game × status multiplier
     │
     ▼
SHAP explanations     ← per-game feature contributions with human-readable labels
     │
     ▼
Ranked recommendations (console summary)
```

---

## Techniques Used

- **Custom sklearn-compatible transformers** (`modules/custom_encoders.py`):
  - `GenreEncoder` — multi-hot encoding for multi-label genres ("Action, RPG" → two flags)
  - `FranchiseEncoder` — James-Stein encoding, shrinking small-sample franchise means toward the global mean
  - `DeveloperEncoder` — M-Estimate encoding for developer reputation
  - `BacklogEncoder` — combines all three plus the numeric features into one transformer
- **Numeric features** — AI-estimated Metacritic and User scores, imputed (median) and standardized in the pipeline
- **Model** — `RidgeCV` with automatic alpha selection over 100 log-spaced candidates
- **Evaluation** — Leave-One-Out Cross-Validation (the training set is tiny — every fold counts), reporting MAE mean, std, and worst-case fold
- **Status multipliers** — a light business rule on top of the model: replays and "give it another chance" games get slightly discounted final scores
- **SHAP explainability** — every prediction carries per-feature SHAP contributions with readable labels, so I know *why* a game ranked where it did
- **Model persistence** — the fitted pipeline is pickled to `data/model.pkl`

### Bonus experiment: TabFM (zero-shot)

[`notebooks/tabfm.ipynb`](notebooks/tabfm.ipynb) reruns the whole problem with [Google's TabFM](https://github.com/google-research/tabfm), a tabular foundation model. No training, no encoders — `fit` just stores the training rows as context and `predict` does in-context learning over them. A fun comparison against the hand-crafted Ridge pipeline.

---

## Project Layout

```
main.py                     entry point — full pipeline run + ranked summary
modules/
  data_ingestion.py         Google Sheets loading + train/backlog split
  custom_encoders.py        the four custom transformers
  pipeline.py               encoder → imputer → scaler → RidgeCV
  train.py                  evaluate, fit, persist
  evaluation.py             LOO-CV metrics
  predict.py                predictions, status multipliers, SHAP breakdowns
notebooks/
  sandbox.ipynb             exploration
  tabfm.ipynb               zero-shot TabFM experiment
data/
  backlog.csv               local cache of the sheet
  model.pkl                 last fitted pipeline
```

---

## Running

```bash
# Install dependencies (requires uv)
uv sync

# Run the full pipeline against the local CSV cache
python main.py
```

By default `main.py` runs with `local=True` (the cached `data/backlog.csv`). To pull live data, call `run(local=False)` — Google Sheets OAuth setup is required, see `modules/data_ingestion.py` for the GCP configuration steps.

---

## Stack

| Tool | Role |
|---|---|
| `scikit-learn` | Pipeline, custom encoders, RidgeCV |
| `category-encoders` | James-Stein and M-Estimate encoding |
| `shap` | Prediction explainability |
| `tabfm` | Zero-shot tabular foundation model experiment |
| `gspread` | Google Sheets integration |
| `pandas` | Data wrangling |
| `uv` | Dependency management |
| `ruff` + `taskipy` | Linting, formatting, task running |
