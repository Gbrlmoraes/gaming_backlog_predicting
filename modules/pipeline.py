import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

from modules.custom_encoders import BacklogEncoder


def build_pipeline(m: int = 10) -> Pipeline:
    """
    BacklogEncoder → RidgeCV

    RidgeCV testa automaticamente vários valores de alpha (força de regularização)
    via CV interno e escolhe o melhor. Com 45 amostras, alphas altos (10–100)
    tendem a vencer — o modelo prefere coeficientes menores e mais estáveis.
    """
    return Pipeline([
        ('enc', BacklogEncoder(m=m)),
        ('model', RidgeCV(alphas=np.logspace(-4, 3, 100))),
    ])
