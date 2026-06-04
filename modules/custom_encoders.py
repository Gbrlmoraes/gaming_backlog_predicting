import pandas as pd
from category_encoders import JamesSteinEncoder, MEstimateEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class GenreEncoder(BaseEstimator, TransformerMixin):
    """
    Gênero → multi-hot encoding.
    """

    def __init__(self, col: str = 'Gênero', sep: str = ', '):
        self.col = col
        self.sep = sep
        self.mlb = MultiLabelBinarizer()

    def _split(self, X: pd.DataFrame):
        return X[self.col].fillna('').str.split(self.sep)

    def fit(self, X: pd.DataFrame, y=None):
        self.mlb.fit(self._split(X))
        self.feature_names_ = list(self.mlb.classes_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = self.mlb.transform(self._split(X))
        return pd.DataFrame(encoded, columns=self.mlb.classes_, index=X.index)


class FranchiseEncoder(BaseEstimator, TransformerMixin):
    """
    Franquia → James-Stein Encoding.
    """

    def __init__(self, col: str = 'Franquia'):
        self.col = col
        self.enc = JamesSteinEncoder(cols=[col], return_df=True)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.enc.fit(X[[self.col]], y)
        self.feature_names_ = [f'{self.col}_js']
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self.enc.transform(X[[self.col]])
        out.columns = self.feature_names_
        out.index = X.index
        return out


class DeveloperEncoder(BaseEstimator, TransformerMixin):
    """
    Desenvolvedora → M-Estimate Encoding.
    """

    def __init__(self, col: str = 'Desenvolvedora', m: int = 10):
        self.col = col
        self.m = m
        self.enc = MEstimateEncoder(cols=[col], m=m, return_df=True)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.enc.fit(X[[self.col]], y)
        self.feature_names_ = [f'{self.col}_mest']
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self.enc.transform(X[[self.col]])
        out.columns = self.feature_names_
        out.index = X.index
        return out


class BacklogEncoder(BaseEstimator, TransformerMixin):
    """
    Combine the three encoders
    """

    def __init__(self, m: int = 10):
        self.m = m
        self.genre_enc = GenreEncoder()
        self.franchise_enc = FranchiseEncoder()
        self.dev_enc = DeveloperEncoder(m=m)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.genre_enc.fit(X)
        self.franchise_enc.fit(X, y)
        self.dev_enc.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        genres = self.genre_enc.transform(X)
        franchise = self.franchise_enc.transform(X)
        developer = self.dev_enc.transform(X)
        return pd.concat([genres, franchise, developer], axis=1)

    def get_feature_names_out(self):
        return (
            self.genre_enc.feature_names_
            + self.franchise_enc.feature_names_
            + self.dev_enc.feature_names_
        )
