import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from modules.data_ingestion import load_sheet, prepare_data
from modules.evaluation import evaluate
from modules.pipeline import build_pipeline

FEATURES: list[str] = ['Gênero', 'Franquia', 'Desenvolvedora']
DEFAULT_MODEL_PATH = Path('data/model.pkl')

warnings.filterwarnings('ignore')


def train(
    sheet_name: str = 'Backlog',
    m: int = 10,
    model_path: Path = DEFAULT_MODEL_PATH,
    verbose: bool = True,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, dict]:
    """
    Full training pipeline: load → evaluate → fit final model → persist.

    Returns the fitted pipeline, training features, training labels,
    and LOO-CV evaluation metrics.
    """
    df = load_sheet(sheet_name)
    finished, backlog = prepare_data(df)

    X_train: pd.DataFrame = finished[FEATURES]
    y_train: pd.Series = finished['Nota']

    pipe = build_pipeline(m=m)

    metrics = evaluate(pipe, X_train, y_train)

    if verbose:
        print('── LOO-CV Evaluation ────────────────────────')
        print(f'  MAE mean  : {metrics["mae_mean"]:.3f}')
        print(f'  MAE std   : {metrics["mae_std"]:.3f}')
        print(f'  Worst case: {metrics["mae_max"]:.3f}')
        print(f'  Folds     : {metrics["n_folds"]}')

    # Fit on full training set
    pipe.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)

    if verbose:
        print(f'\nModel saved → {model_path}')

    return pipe, X_train, y_train, metrics


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
