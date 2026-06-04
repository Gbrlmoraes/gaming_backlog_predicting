import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Leave-One-Out Cross-Validation.
    When the backlog data grows, maybe change to k-folds
    """
    loo = LeaveOneOut()
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=loo,
        scoring='neg_mean_absolute_error',
    )
    mae_per_fold = -scores

    return {
        'mae_mean': mae_per_fold.mean(),
        'mae_std': mae_per_fold.std(),
        'mae_max': mae_per_fold.max(),
        'n_folds': len(scores),
    }
