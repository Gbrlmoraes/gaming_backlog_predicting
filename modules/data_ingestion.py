import os

import gspread
import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')


def load_sheet(sheet_name: str = 'Backlog', local: bool = False) -> pd.DataFrame:
    """
    Loads backlog sheet from Google Sheets.
    Steps:
        # 1. Enable Sheets and Drive API in GCP concole
        # 2. Create OAuth JSON in GCP console
        # 3. Add personal email as test user
        # 4. Authorize application in the browser
    """
    if not local:
        gc = gspread.oauth()
        sh = gc.open(sheet_name)
        worksheet = sh.get_worksheet(0)
        df = pd.DataFrame(worksheet.get_all_records())

        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(
            os.path.join(DATA_DIR, 'backlog'),
            encoding='utf8',
            sep=';',
            decimal=',',
            index=False,
        )

    else:
        df = pd.read_csv(
            os.path.join(DATA_DIR, 'backlog'),
            encoding='utf8',
            sep=';',
            decimal=',',
        )

    return df


def prepare_data(df: pd.DataFrame):
    """
    Splits finished games (train) and backlog games (prediction).
    """
    df = df.copy()
    df['Franquia'] = df['Franquia'].fillna('Independente').replace('—', 'Independente')
    df['Desenvolvedora'] = (
        df['Desenvolvedora'].fillna('Desconhecida').replace('—', 'Desconhecida')
    )
    df['Nota'] = pd.to_numeric(df['Nota'], errors='coerce')

    finished = (
        df[df['Status'] == '3. Finalizados']
        .dropna(subset=['Nota'])
        .reset_index(drop=True)
    )

    backlog = df[
        df['Status'].isin([
            '2. Próximos',
            '4. Backlog',
            '5. Rejogar',
            '6. Dar Outra Chance',
        ])
    ].reset_index(drop=True)

    return finished, backlog
