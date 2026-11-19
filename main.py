"""Pipeline completo: dados -> split -> modelo -> metricas -> threshold -> PDF."""
import json

from src import config as C
from src.data import generate_dataset, split_data
from src.metrics import evaluate
from src.model import build_baseline, build_model
from src.plots import plot_accuracy_trap, plot_cost, plot_roc
from src.report import build_pdf
from src.threshold import best_threshold, cost_curve, scenarios, total_cost

import pandas as pd


def run():
    C.REPORTS.mkdir(exist_ok=True)
    df = generate_dataset()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)

    base = build_baseline().fit(X_tr, y_tr)
    model = build_model().fit(X_tr, y_tr)          # scaler aprende SO no treino
    s_te_b = base.predict_proba(X_te)[:, 1]
    s_va, s_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]

    curve_va = cost_curve(y_va, s_va)
    t_star = best_threshold(curve_va)               # escolhido na VALIDACAO
    curve_te = cost_curve(y_te, s_te)

    table = pd.DataFrame({
        "Baseline (sempre OK)": evaluate(y_te, s_te_b, 0.5),
        "Modelo @ 0,5": evaluate(y_te, s_te, C.DEFAULT_THRESHOLD),
        f"Modelo @ {t_star:.3f}": evaluate(y_te, s_te, t_star),
    }).T
    c = {k: total_cost(table.loc[k, "FN"], table.loc[k, "FP"]) for k in table.index}
    table["Custo (R$)"] = pd.Series(c)
    sc = scenarios(y_va, s_va, y_te, s_te)

    plot_cost(curve_te, curve_va, t_star, C.DEFAULT_THRESHOLD, C.REPORTS / "custo_threshold.png")
    plot_roc(y_te, s_te, C.REPORTS / "roc.png")
    plot_accuracy_trap(table, C.REPORTS / "armadilha_acuracia.png")
    table.to_csv(C.REPORTS / "metricas.csv"); sc.to_csv(C.REPORTS / "cenarios.csv", index=False)

    info = {"n": len(df), "taxa_falha": float(df.falha.mean()),
            "n_treino": len(X_tr), "n_val": len(X_va), "n_teste": len(X_te),
            "falhas_teste": int(y_te.sum()), "t_star": t_star,
            "economia": float(c["Modelo @ 0,5"] - c[f"Modelo @ {t_star:.3f}"])}
    (C.REPORTS / "resumo.json").write_text(json.dumps(info, indent=2))
    build_pdf(table, sc, info, C.REPORTS / "Relatorio_Executivo.pdf")
    print(table.round(4).to_string()); print(sc.round(3).to_string()); print(info)


if __name__ == "__main__":
    run()
