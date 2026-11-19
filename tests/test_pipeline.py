import numpy as np
from src import config as C
from src.data import generate_dataset, split_data
from src.metrics import evaluate, fbeta


def test_reprodutivel_e_desbalanceado():
    a, b = generate_dataset(20000), generate_dataset(20000)
    assert a.equals(b)
    assert abs(a.falha.mean() - C.FAULT_RATE) < 0.001


def test_split_sem_sobreposicao_e_estratificado():
    df = generate_dataset(20000)
    Xtr, Xva, Xte, ytr, yva, yte = split_data(df)
    assert not (set(Xtr.index) & set(Xte.index)) and not (set(Xva.index) & set(Xte.index))
    assert abs(yte.mean() - ytr.mean()) < 0.001


def test_metricas_conhecidas():
    y = np.array([1, 1, 0, 0, 0, 0]); s = np.array([.9, .2, .8, .1, .1, .1])
    m = evaluate(y, s, 0.5)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 3)
    assert m["Precisao"] == 0.5 and m["Recall"] == 0.5 and abs(m["F1"] - 0.5) < 1e-9
    assert abs(fbeta(0.5, 1.0, 2) - 5 * 0.5 / (4 * 0.5 + 1)) < 1e-9
