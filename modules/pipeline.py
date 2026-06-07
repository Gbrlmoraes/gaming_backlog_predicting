import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.custom_encoders import BacklogEncoder


def build_pipeline(m: int = 10) -> Pipeline:
    """
    BacklogEncoder → SimpleImputer → StandardScaler → RidgeCV
    """
    return Pipeline([
        ('enc', BacklogEncoder(m=m)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=np.logspace(-4, 3, 100))),
    ])
