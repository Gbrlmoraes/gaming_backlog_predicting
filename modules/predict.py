from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

FEATURES: list[str] = [
    'Gênero',
    'Franquia',
    'Desenvolvedora',
    'Metacritic Score (AI)',
    'User Score (AI)',
]

STATUS_MULTIPLIERS: dict[str, float] = {
    '2. Próximos': 1.00,
    '4. Backlog': 1.00,
    '5. Rejogar': 0.95,
    '6. Dar Outra Chance': 0.87,
}


@dataclass
class ShapContribution:
    """SHAP contribution for a single feature of a single game."""

    feature: str
    encoded_value: float
    shap_value: float
    human_label: str


@dataclass
class GamePrediction:
    """Full prediction result with SHAP breakdown for one game."""

    game: str
    franchise: str
    developer: str
    genre: str
    status: str
    base_score: float
    status_multiplier: float
    final_score: float
    shap_base_value: float
    shap_contributions: list[ShapContribution] = field(default_factory=list)

    @property
    def top_contributions(self) -> list[ShapContribution]:
        """Contributions sorted by absolute SHAP value (descending)."""
        return sorted(
            self.shap_contributions, key=lambda c: abs(c.shap_value), reverse=True
        )


def _get_global_means(encoder, X_train: pd.DataFrame) -> dict[str, float]:
    """Return global means of encoded franchise and developer columns."""
    X_enc: pd.DataFrame = encoder.transform(X_train)
    return {
        'Franquia_js': float(X_enc['Franquia_js'].mean()),
        'Desenvolvedora_mest': float(X_enc['Desenvolvedora_mest'].mean()),
    }


def _format_category_label(
    prefix: str,
    name: str,
    encoded_value: float,
    global_mean: float,
) -> str:
    """
    Format a human-readable label for a target-encoded feature.

    name is empty string when the category was unseen at training time
    (encoder falls back to global mean).
    """
    if name:
        return f'{prefix} ({name}, mean={encoded_value:.2f})'
    return f'{prefix} (unknown — global mean={global_mean:.2f})'


def _build_shap_explainer(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[shap.Explainer, object, list[str]]:
    """
    Fit the encoder on training data and construct a SHAP LinearExplainer.

    The explainer is built on the fully pre-processed data (enc → imputer → scaler)
    so that SHAP values are consistent with what the Ridge model actually saw.
    The raw encoder output (pre-imputer) is returned separately for human-readable labels.

    Returns the explainer, the fitted encoder step, and the encoded feature names.
    """
    encoder = pipe.named_steps['enc']
    encoder.fit(X_train, y_train)

    X_enc_train: pd.DataFrame = encoder.transform(X_train)
    feature_names: list[str] = list(X_enc_train.columns)

    # Apply imputer and scaler so the explainer sees the same space as the Ridge
    X_processed_train = pipe.named_steps['imputer'].transform(X_enc_train)
    X_processed_train = pipe.named_steps['scaler'].transform(X_processed_train)

    ridge = pipe.named_steps['model']
    explainer = shap.LinearExplainer(ridge, X_processed_train)

    return explainer, encoder, feature_names


def _compute_shap_values(
    explainer: shap.Explainer,
    encoder,
    pipe: Pipeline,
    feature_names: list[str],
    X_backlog: pd.DataFrame,
) -> shap.Explanation:
    """Encode the backlog features and compute SHAP values."""
    X_enc_backlog: pd.DataFrame = encoder.transform(X_backlog)
    # Store raw encoded values for human-readable display before scaling
    raw_encoded_values = X_enc_backlog.values.copy()

    X_processed = pipe.named_steps['imputer'].transform(X_enc_backlog)
    X_processed = pipe.named_steps['scaler'].transform(X_processed)

    raw = explainer(X_processed)
    return shap.Explanation(
        values=raw.values,
        base_values=raw.base_values,
        data=raw_encoded_values,  # pre-scale values for readable SHAP display
        feature_names=feature_names,
    )


def predict_backlog(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    backlog: pd.DataFrame,
    status_multipliers: dict[str, float] | None = None,
    top_n_shap: int = 5,
) -> list[GamePrediction]:
    """
    Predict ratings for all backlog games and attach SHAP explanations.

    Every returned GamePrediction contains the full ranked list of
    ShapContribution objects so callers can inspect any depth of explanation.

    Args:
        pipe:               Fitted sklearn Pipeline (BacklogEncoder → RidgeCV).
        X_train:            Feature DataFrame used to train the model.
        y_train:            Target series used to train the model.
        backlog:            DataFrame of games to predict (must include FEATURES cols).
        status_multipliers: Post-prediction multipliers by game status.
        top_n_shap:         How many top contributions to surface (all are stored).

    Returns:
        List of GamePrediction, sorted by final_score descending.
    """
    if status_multipliers is None:
        status_multipliers = STATUS_MULTIPLIERS

    # Build SHAP infrastructure
    explainer, encoder, feature_names = _build_shap_explainer(pipe, X_train, y_train)
    global_means = _get_global_means(encoder, X_train)

    # Raw predictions
    base_scores: np.ndarray = pipe.predict(backlog[FEATURES])

    # SHAP values for every backlog game
    shap_values = _compute_shap_values(
        explainer, encoder, pipe, feature_names, backlog[FEATURES]
    )

    results: list[GamePrediction] = []

    for i, (_, row) in enumerate(backlog.reset_index(drop=True).iterrows()):
        multiplier = status_multipliers.get(row['Status'], 1.0)
        sv = shap_values[i]

        contributions: list[ShapContribution] = []
        for feat, enc_val, shap_val in zip(
            sv.feature_names,
            sv.data,
            sv.values,
        ):
            if feat == 'Franquia_js':
                human = _format_category_label(
                    'Franquia',
                    row['Franquia'],
                    enc_val,
                    global_means['Franquia_js'],
                )
            elif feat == 'Desenvolvedora_mest':
                human = _format_category_label(
                    'Desenvolvedora',
                    row['Desenvolvedora'],
                    enc_val,
                    global_means['Desenvolvedora_mest'],
                )
            elif feat in ('Metacritic Score (AI)', 'User Score (AI)'):
                label = feat
                human = (
                    f'{label} ({enc_val:.1f})'
                    if not np.isnan(enc_val)
                    else f'{label} (N/A)'
                )
            else:
                # Genre multi-hot: show presence/absence
                human = feat if enc_val == 1 else f'{feat} (absent)'

            contributions.append(
                ShapContribution(
                    feature=feat,
                    encoded_value=float(enc_val),
                    shap_value=float(shap_val),
                    human_label=human,
                )
            )

        results.append(
            GamePrediction(
                game=row['Jogo'],
                franchise=row['Franquia'],
                developer=row['Desenvolvedora'],
                genre=row['Gênero'],
                status=row['Status'],
                base_score=round(float(base_scores[i]), 4),
                status_multiplier=multiplier,
                final_score=round(float(base_scores[i]) * multiplier, 4),
                shap_base_value=round(float(sv.base_values), 4),
                shap_contributions=contributions,
            )
        )

    results.sort(key=lambda g: g.final_score, reverse=True)
    return results


def predictions_to_dataframe(
    predictions: list[GamePrediction],
    top_shap: int = 3,
) -> pd.DataFrame:
    """
    Convert a list of GamePrediction objects to a readable DataFrame.

    Each row is one game; top-N SHAP contributions appear as separate columns.
    """
    rows = []
    for pred in predictions:
        row: dict = {
            'game': pred.game,
            'franchise': pred.franchise,
            'status': pred.status,
            'shap_base': pred.shap_base_value,
            'base_score': pred.base_score,
            'multiplier': pred.status_multiplier,
            'final_score': pred.final_score,
        }
        for rank, contrib in enumerate(pred.top_contributions[:top_shap], start=1):
            sign = '+' if contrib.shap_value >= 0 else ''
            row[f'shap_top{rank}'] = (
                f'{contrib.human_label}: {sign}{contrib.shap_value:.3f}'
            )
        rows.append(row)

    return pd.DataFrame(rows)
