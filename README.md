# Gaming Backlog Predictor

A personal project to predict which game I should play next — built just for fun, but using real MLOps practices: modular code, proper evaluation, feature engineering, and model persistence.

---

## What Was Built

A regression pipeline that reads my gaming backlog from Google Sheets, trains on finished games (using my personal ratings as the target), and ranks unplayed games by predicted score.

**Key components:**

- **Data ingestion** — pulls data live from Google Sheets via the GCP API and caches it locally as CSV
- **Feature engineering** — three custom `sklearn`-compatible transformers:
  - `GenreEncoder`: multi-hot encoding for multi-label genres
  - `FranchiseEncoder`: James-Stein encoding for franchise identity
  - `DeveloperEncoder`: M-Estimate encoding for developer reputation
- **Model** — `RidgeCV` with automatic alpha selection via internal cross-validation (100 log-spaced candidates)
- **Evaluation** — Leave-One-Out Cross-Validation reporting MAE mean, std, and worst-case fold
- **SHAP explanations** — every prediction includes per-feature SHAP contributions with human-readable labels, so you know *why* a game was ranked highly
- **Model persistence** — fitted pipeline serialized to `data/model.pkl`

---

## Pipeline

```
Google Sheets
     │
     ▼
load_sheet()          ← pulls live data via gspread OAuth
     │
     ▼
prepare_data()        ← splits into finished (train) / backlog (infer)
     │
     ▼
BacklogEncoder        ← multi-hot genres + James-Stein franchise + M-Estimate developer
     │
     ▼
RidgeCV               ← LOO-CV evaluation → fit on full training set → save model
     │
     ▼
predict_backlog()     ← scores every backlog game
     │
     ▼
SHAP explanations     ← per-game feature contributions
     │
     ▼
Ranked recommendations (console summary)
```

---

## TODO

- [ ] Build a UI (web app or simple dashboard) to browse recommendations interactively
- [ ] Improve model training — experiment with other regressors, hyperparameter tuning, ensemble methods
- [ ] Improve feature engineering — add external data (Metacritic scores, release year, playtime estimates)

---

## Running

```bash
# Install dependencies (requires uv)
uv sync

# Run the full pipeline
python main.py
```

Google Sheets OAuth setup required — see `modules/data_ingestion.py` for the GCP configuration steps.

---

## Stack

| Tool | Role |
|---|---|
| `scikit-learn` | Pipeline, encoders, RidgeCV |
| `category-encoders` | James-Stein and M-Estimate encoding |
| `shap` | Prediction explainability |
| `gspread` | Google Sheets integration |
| `pandas` | Data wrangling |
| `uv` | Dependency management |
| `ruff` | Linting and formatting |
