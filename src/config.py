"""Parametros centrais do projeto (fonte unica de verdade)."""
from pathlib import Path

SEED = 42
N_SAMPLES = 200_000
FAULT_RATE = 0.005          # 0,5 % de falhas (99,5 % operacional)
TEST_SIZE = 0.20            # isolado ANTES de qualquer pre-processamento
VAL_SIZE = 0.25             # fatia do treino usada para escolher o threshold

FEATURES = ["vibracao_rms", "temperatura_motor", "corrente_A",
            "pressao_hidraulica", "rpm", "ruido_acustico_db"]

# Premissas financeiras (R$) - ajustaveis
COST_FN = 50_000.0          # falha nao detectada: parada nao planejada de linha
COST_FP = 500.0             # alarme falso: inspecao/parada preventiva desnecessaria
DEFAULT_THRESHOLD = 0.5

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
