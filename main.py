from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.data_ingestion import load_sheet, prepare_data
from modules.predict import GamePrediction, predict_backlog, predictions_to_dataframe
from modules.train import DEFAULT_MODEL_PATH, FEATURES, train

SHEET_NAME = 'Backlog'
M_ESTIMATE_PARAM = 10
MODEL_PATH = DEFAULT_MODEL_PATH
TOP_SHAP_COLUMNS = 3  # columns shown in the summary DataFrame
TOP_GAMES_PRINT = 15  # rows printed in the console summary


def run(
    sheet_name: str = SHEET_NAME,
    m: int = M_ESTIMATE_PARAM,
    model_path: Path = MODEL_PATH,
    top_shap: int = TOP_SHAP_COLUMNS,
    verbose: bool = True,
) -> list[GamePrediction]:
    """
    Run the complete pipeline and return all predictions with SHAP explanations.

    Steps:
        1. Load data from Google Sheets.
        2. Split into finished games (train) and backlog (infer).
        3. Evaluate via LOO-CV, then fit the final model.
        4. Predict ratings for every backlog game.
        5. Attach SHAP contributions to each prediction.
        6. Print a ranked summary table.

    Returns:
        Sorted list of GamePrediction (best → worst final_score), each carrying
        the full list of ShapContribution objects for downstream use.
    """
    if verbose:
        print('Loading data from Google Sheets…')

    df = load_sheet(sheet_name)
    finished, backlog = prepare_data(df)

    X_train = finished[FEATURES]
    y_train = finished['Nota']

    if verbose:
        print(f'  Finished games (train): {len(finished)}')
        print(f'  Backlog games (infer) : {len(backlog)}\n')

    pipe, _, _, metrics = train(
        sheet_name=sheet_name,
        m=m,
        model_path=model_path,
        verbose=verbose,
    )

    if verbose:
        print('\nComputing predictions and SHAP explanations…')

    predictions = predict_backlog(
        pipe=pipe,
        X_train=X_train,
        y_train=y_train,
        backlog=backlog,
    )

    if verbose:
        summary: pd.DataFrame = predictions_to_dataframe(predictions, top_shap=top_shap)

        print(f'\n── Top {TOP_GAMES_PRINT} recommended games ─────────────────────────')
        print(summary.head(TOP_GAMES_PRINT).to_string(index=False))

        print('\n── SHAP detail for #1 recommendation ────────────────────')
        top = predictions[0]
        print(f'  Game      : {top.game}')
        print(f'  Base score: {top.shap_base_value:.3f}  (average of training ratings)')
        print(f'  Final score: {top.final_score:.3f}  (base × {top.status_multiplier})')
        print()
        for contrib in top.top_contributions[:6]:
            sign = '+' if contrib.shap_value >= 0 else ''
            print(f'    {sign}{contrib.shap_value:+.3f}  {contrib.human_label}')

    return predictions


if __name__ == '__main__':
    all_predictions = run()

    # The full list is available for downstream processing.
    # Example: access every prediction's SHAP breakdown programmatically.
    for pred in all_predictions:
        top3 = pred.top_contributions[:3]
        labels = ' | '.join(f'{c.human_label}: {c.shap_value:+.3f}' for c in top3)
        print(f'{pred.final_score:.2f}  {pred.game:<35} {labels}')
