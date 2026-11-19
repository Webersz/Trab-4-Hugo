"""Relatorio Executivo em PDF (reportlab)."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config as C


def brl(v):
    return "R$ " + f"{v:,.0f}".replace(",", ".")


def _tbl(data, widths=None, fs=7.5):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), fs),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    return t


def build_pdf(table, sc, info, path):
    ss = getSampleStyleSheet(); H, B = ss["Heading2"], ss["BodyText"]; B.alignment = 4
    S = []
    add = S.append
    n = f"{info['n']:,}".replace(",", ".")
    add(Paragraph("Avaliação de Métricas em Modelos de Classificação Desbalanceados", ss["Title"]))
    add(Paragraph("Relatório Executivo - Engenharia de Dados e MLOps - UniSENAI", ss["Heading4"]))
    add(Paragraph('Repositório Git: <link href="https://github.com/Webersz/Trab-4-Hugo" color="blue">'
                  'https://github.com/Webersz/Trab-4-Hugo</link>', ss["BodyText"]))
    t_star, eco = info["t_star"], info["economia"]
    add(Paragraph(
        f"<b>Resumo:</b> em um conjunto de {n} leituras de sensores com {info['taxa_falha']*100:.2f}% de falhas, "
        f"um modelo que nunca detecta falha atinge {table.iloc[0]['Acuracia']*100:.2f}% de acurácia. "
        f"O modelo logístico com threshold padrão (0,5) tem recall de {table.iloc[1]['Recall']*100:.1f}%. "
        f"Ajustando o threshold para {t_star:.3f} (escolhido por custo, em validação), a economia no conjunto de teste é de "
        f"<b>{brl(eco)}</b> em relação ao threshold padrão.", B))

    add(Paragraph("1. Arquitetura da Solução e prevenção de Data Leakage", H))
    f = lambda x: f"{x:,}".replace(",", ".")
    add(Paragraph(
        f"Os dados sintéticos (6 sensores: vibração, temperatura, corrente, pressão, rpm, ruído; semente fixa {C.SEED}) são "
        f"gerados de forma reprodutível. <b>O conjunto de teste ({f(info['n_teste'])} linhas, {info['falhas_teste']} falhas) é isolado "
        "antes de qualquer pré-processamento</b>, com divisão estratificada para preservar a taxa de 0,5%. Do restante, uma "
        f"fatia de validação ({f(info['n_val'])} linhas) é separada para escolher o threshold; o treino tem {f(info['n_treino'])} linhas. "
        "A padronização (StandardScaler) e o modelo vivem em um único <i>Pipeline</i> ajustado somente no treino: média e desvio "
        "do teste nunca influenciam o ajuste. O threshold é escolhido na validação e apenas <i>avaliado</i> no teste, "
        "evitando otimismo por ajuste ao conjunto de teste. Fluxo: gerar &rarr; split (teste primeiro) &rarr; fit no treino &rarr; "
        "threshold na validação &rarr; avaliação única no teste.", B))

    add(Paragraph("2. Quadro Comparativo de Métricas (conjunto de teste)", H))
    cols = ["TP", "TN", "FP", "FN", "Acuracia", "Precisao", "Recall", "F1", "F2", "F0.5", "AUC-ROC"]
    names = ["TP", "TN", "FP", "FN", "Acurácia", "Precisão", "Recall", "F1", "F2", "F0.5", "AUC-ROC"]
    rows = [["Cenário"] + names]
    for k, r in table.iterrows():
        rows.append([k] + [str(int(r[c])) if c in ("TP", "TN", "FP", "FN") else f"{r[c]:.4f}" for c in cols])
    add(_tbl(rows, [3.2 * cm] + [1.25 * cm] * 4 + [1.35 * cm] * 7, fs=6.5))
    add(Spacer(1, 6))
    add(Paragraph(
        "<b>Leitura:</b> a acurácia é alta em todos os casos (a classe majoritária domina), logo não discrimina modelos. "
        "Precisão mede quantos alarmes são reais; Recall mede quantas falhas são capturadas. F1 pesa ambos igualmente, "
        "F2 privilegia recall (adequado quando FN é muito mais caro) e F0.5 privilegia precisão. O baseline tem AUC 0,5 e "
        "recall zero. A AUC-ROC independe do threshold: mede a capacidade de ranqueamento, enquanto o threshold define o "
        "ponto de operação.", B))
    add(Image(str(C.REPORTS / "armadilha_acuracia.png"), 14 * cm, 6.6 * cm))
    add(PageBreak())

    add(Paragraph("3. Análise Financeira do Threshold", H))
    add(Paragraph(
        f"Premissas (ajustáveis em config.py): falha não detectada (FN) = {brl(C.COST_FN)} (parada não planejada); alarme falso (FP) = "
        f"{brl(C.COST_FP)} (inspeção). Custo total = FN&times;{brl(C.COST_FN)} + FP&times;{brl(C.COST_FP)}. O gráfico mostra o custo no teste "
        "em função do threshold; o ponto verde é o threshold escolhido na validação e o vermelho, o padrão 0,5.", B))
    add(Image(str(C.REPORTS / "custo_threshold.png"), 15 * cm, 7.9 * cm))
    c0, c1 = table.iloc[1]["Custo (R$)"], table.iloc[2]["Custo (R$)"]
    add(Paragraph(f"<b>Resultado:</b> custo @0,5 = {brl(c0)}; custo @{t_star:.3f} = {brl(c1)}; "
                  f"<b>economia = {brl(eco)} ({eco/c0*100:.1f}%)</b> no período de teste.", B))
    add(Spacer(1, 6))
    add(Paragraph("Sensibilidade a cenários operacionais (threshold ótimo recalculado na validação):", B))
    rows = [["Cenário", "Custo FN", "Custo FP", "Threshold*", "Custo @0,5", "Custo @ótimo", "Economia"]]
    for _, r in sc.iterrows():
        rows.append([r["Cenario"], brl(r["Custo FN"]), brl(r["Custo FP"]), f"{r['Threshold*']:.3f}",
                     brl(r["Custo @0,5"]), brl(r["Custo @otimo"]), brl(r["Economia"])])
    add(_tbl(rows, [4.6 * cm, 2 * cm, 1.8 * cm, 1.8 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm], fs=7))
    add(Spacer(1, 6))
    add(Paragraph("Quanto maior a razão custo(FN)/custo(FP), menor o threshold ótimo (mais alarmes aceitos). "
                  "Obs.: com apenas ~200 falhas no teste, o custo é ruidoso; o resultado é indicativo, não garantido.", B))
    add(Image(str(C.REPORTS / "roc.png"), 8 * cm, 7.5 * cm))

    add(Paragraph("4. Recomendações MLOps (monitoramento em produção)", H))
    for t in [
        "<b>Versionamento:</b> versionar código (Git), dados e modelo (ex.: DVC/MLflow Registry) com a semente e os parâmetros de cada execução.",
        "<b>Métricas de negócio:</b> acompanhar recall, precisão, F2 e custo estimado por semana - nunca apenas acurácia.",
        "<b>Drift:</b> monitorar distribuição dos sensores (PSI/KS) e das probabilidades previstas; alertar se PSI &gt; 0,2.",
        "<b>Rótulos tardios:</b> registrar falhas reais e alarmes para recalcular a matriz de confusão com defasagem (ex.: 30 dias).",
        "<b>Threshold como configuração:</b> reavaliar o threshold ao mudar custos (FN/FP) ou o modelo; guardá-lo fora do código.",
        "<b>Retreino e rollout:</b> retreinar por gatilho (queda de recall/drift) ou periodicamente; usar shadow/canary antes de promover.",
        "<b>Operação:</b> alerta se a taxa de alarmes sair de uma faixa esperada; fallback para regra de engenharia se o serviço falhar; testes automatizados (pytest) no CI.",
    ]:
        add(Paragraph("&bull; " + t, B))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Relatorio Executivo").build(S)
