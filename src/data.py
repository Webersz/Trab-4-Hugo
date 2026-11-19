"""Geracao reprodutivel de dados sinteticos de sensores e split sem leakage."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from . import config as C


def generate_dataset(n=C.N_SAMPLES, fault_rate=C.FAULT_RATE, seed=C.SEED) -> pd.DataFrame:
    """Gera sensores industriais; y=1 (falha) ~0,5 %. Classes parcialmente sobrepostas."""
    X, y = make_classification(
        n_samples=n, n_features=len(C.FEATURES), n_informative=4, n_redundant=1,
        n_repeated=0, n_classes=2, weights=[1 - fault_rate, fault_rate],
        n_clusters_per_class=1, class_sep=1.0, flip_y=0.0, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    # escalas fisicas plausiveis (transformacao afim -> nao altera separabilidade)
    loc = np.array([2.5, 70.0, 12.0, 150.0, 1800.0, 65.0])
    scale = np.array([0.8, 8.0, 2.0, 15.0, 120.0, 5.0])
    X = loc + X * scale + rng.normal(0, 0.01, X.shape) * scale
    df = pd.DataFrame(X, columns=C.FEATURES)
    df["falha"] = y
    return df


def split_data(df: pd.DataFrame, seed=C.SEED):
    """Isola o TESTE primeiro (estratificado); depois separa validacao do treino.

    Nenhuma estatistica (media, desvio, etc.) e calculada antes deste passo,
    evitando Data Leakage. O scaler e ajustado apenas no treino (ver model.py).
    """
    X, y = df[C.FEATURES], df["falha"]
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=C.VAL_SIZE, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te
