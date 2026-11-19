"""Graficos do relatorio."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from . import config as C


def plot_cost(curve_te, curve_va, t_star, t_def, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(curve_te["threshold"], curve_te["custo"] / 1e3, label="Custo total (teste)", color="#1f4e79")
    ax.plot(curve_va["threshold"], curve_va["custo"] / 1e3, label="Custo total (validacao)",
            color="#999999", ls="--", lw=1)
    c_def = curve_te.iloc[(curve_te["threshold"] - t_def).abs().argmin()]["custo"] / 1e3
    c_opt = curve_te.iloc[(curve_te["threshold"] - t_star).abs().argmin()]["custo"] / 1e3
    ax.axvline(t_def, color="#c0392b", ls=":"); ax.scatter([t_def], [c_def], color="#c0392b", zorder=5,
        label=f"Padrao 0,5: R$ {c_def*1e3:,.0f}")
    ax.axvline(t_star, color="#27ae60", ls=":"); ax.scatter([t_star], [c_opt], color="#27ae60", zorder=5,
        label=f"Otimo {t_star:.3f}: R$ {c_opt*1e3:,.0f}")
    ax.set_xlabel("Threshold de decisao"); ax.set_ylabel("Custo total (R$ mil)")
    ax.set_title(r"Custo total vs. threshold (FN = R\$ 50.000 | FP = R\$ 500)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_te, s_te, path):
    fpr, tpr, _ = roc_curve(y_te, s_te)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, color="#1f4e79", label="Regressao Logistica"); ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("Taxa de FP"); ax.set_ylabel("Recall (TPR)"); ax.set_title("Curva ROC (teste)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_accuracy_trap(df_metrics, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = ["Acuracia", "Recall", "Precisao", "F2"]
    df_metrics[cols].plot.bar(ax=ax, rot=0, color=["#7f8c8d", "#c0392b", "#2980b9", "#27ae60"])
    ax.set_ylim(0, 1.05); ax.set_title("Armadilha da acuracia"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
