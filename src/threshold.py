"""Ajuste dinamico do threshold por minimizacao do custo financeiro esperado."""
import numpy as np
import pandas as pd

from . import config as C
from .metrics import confusion


def total_cost(fn, fp, cost_fn=C.COST_FN, cost_fp=C.COST_FP) -> float:
    return fn * cost_fn + fp * cost_fp


def cost_curve(y_true, y_score, cost_fn=C.COST_FN, cost_fp=C.COST_FP, grid=None) -> pd.DataFrame:
    grid = np.linspace(0.001, 0.999, 999) if grid is None else grid
    rows = []
    for t in grid:
        cm = confusion(y_true, (y_score >= t).astype(int))
        rows.append({"threshold": t, **cm, "custo": total_cost(cm["FN"], cm["FP"], cost_fn, cost_fp)})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    """Menor custo; em empate escolhe o threshold mediano entre os otimos (mais estavel)."""
    best = curve[curve["custo"] == curve["custo"].min()]["threshold"]
    return float(best.median())


def scenarios(y_val, s_val, y_te, s_te):
    """Cenarios operacionais: (nome, custo_FN, custo_FP). Threshold escolhido na VALIDACAO."""
    cenarios = [("Linha critica (FN muito caro)", 100_000.0, 500.0),
                ("Base (premissa do relatorio)", C.COST_FN, C.COST_FP),
                ("Inspecao cara (FP caro)", 20_000.0, 2_000.0)]
    out = []
    for nome, cfn, cfp in cenarios:
        t = best_threshold(cost_curve(y_val, s_val, cfn, cfp))
        cm = confusion(y_te, (s_te >= t).astype(int))
        cm0 = confusion(y_te, (s_te >= C.DEFAULT_THRESHOLD).astype(int))
        c_opt, c_def = total_cost(cm["FN"], cm["FP"], cfn, cfp), total_cost(cm0["FN"], cm0["FP"], cfn, cfp)
        out.append({"Cenario": nome, "Custo FN": cfn, "Custo FP": cfp, "Threshold*": t,
                    "Custo @0,5": c_def, "Custo @otimo": c_opt, "Economia": c_def - c_opt})
    return pd.DataFrame(out)
