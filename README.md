# Avaliação de Métricas em Modelos de Classificação Desbalanceados

Trabalho Prático – Engenharia de Dados e MLOps (UniSENAI).
Detecção de falhas raras (0,5 %) em sensores industriais sintéticos, comparando métricas
(Acurácia, Matriz de Confusão, Precisão, Recall, F1/F2/F0.5, AUC-ROC) e ajustando o
threshold de decisão por **custo financeiro**.

## Estrutura

```
main.py            # executa todo o pipeline e gera o PDF
src/config.py      # parâmetros (semente, taxa de falha, custos FN/FP)
src/data.py        # geração de dados + split (teste isolado primeiro)
src/model.py       # Pipeline StandardScaler + Regressão Logística; baseline
src/metrics.py     # métricas a partir de TP/TN/FP/FN
src/threshold.py   # curva de custo, threshold ótimo, cenários
src/plots.py       # gráficos
src/report.py      # relatório PDF (reportlab)
tests/             # testes pytest
reports/           # saídas: PDF, PNGs, CSVs
.vscode/           # configuração do VSCode
```

## Ambiente (VSCode)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Abra a pasta no VSCode, instale as extensões recomendadas (`.vscode/extensions.json`) e
selecione o interpretador `.venv`.

## Execução

```bash
python main.py          # gera reports/Relatorio_Executivo.pdf
pytest -q               # testes
```

## Prevenção de Data Leakage

1. O conjunto de **teste (20 %) é separado primeiro**, estratificado.
2. Do restante, 25 % vira **validação** (usada só para escolher o threshold).
3. `StandardScaler` + modelo ficam em um `Pipeline` ajustado **somente no treino**.
4. O teste é usado uma única vez, para avaliação final.

## Premissas financeiras (editáveis em `src/config.py`)

- Falha não detectada (FN): R$ 50.000 (parada não planejada)
- Alarme falso (FP): R$ 500 (inspeção)

Esses valores são hipóteses didáticas, não dados reais de uma planta.

## Resultados (semente 42)

| Cenário | FN | FP | Recall | Custo |
|---|---|---|---|---|
| Baseline (sempre OK) | 200 | 0 | 0,00 | R$ 10.000.000 |
| Modelo @ 0,5 | 46 | 0 | 0,77 | R$ 2.300.000 |
| Modelo @ 0,026 (ótimo) | 20 | 179 | 0,90 | R$ 1.089.500 |

Economia vs. threshold padrão: **R$ 1.210.500** no conjunto de teste.
