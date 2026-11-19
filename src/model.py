"""Modelo: Pipeline (StandardScaler + Regressao Logistica) - scaler aprende so no treino."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def build_model() -> Pipeline:
    # Sem class_weight de proposito: mostra a falha do threshold 0,5 em dados raros.
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=C.SEED))])


def build_baseline() -> DummyClassifier:
    """Baseline 'sempre operacional' - ilustra a armadilha da acuracia."""
    return DummyClassifier(strategy="most_frequent")
