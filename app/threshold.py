"""
Limiar Adaptativo — mecanismo de ajuste do threshold de validação.

Design:
  threshold_final = BASE_THRESHOLD × specialty_factor × quality_factor

Justificativa clínica dos ajustes por especialidade:
  - Oncologia (+10%): laudos oncológicos frequentemente chegam fora do padrão
    esperado (laudos de outros países, terminologias variadas). Um falso positivo
    aqui pode resultar em tomada de decisão clínica grave com base em documento
    inválido.
  - Psiquiatria (+5%): documentos de saúde mental têm formato menos padronizado.
    Aumentar o rigor reduz o risco de aceitar um relato informal como laudo formal.
  - Cardiologia (+5%): exames cardíacos digitalizados frequentemente têm layouts
    complexos que confundem o OCR. Melhor rejeitar e pedir reenvio.
  - Clínica geral (0%): base de comparação.
  - Pediatria (+5%): contexto de vulnerabilidade — rigor adicional é justificado.

Ajuste por qualidade de extração:
  - OCR em PDF escaneado é menos confiável → +5% no limiar.
  - Texto muito curto (< 30 palavras) → +10% (sinal de extração falha).

Histórico de erros (em memória — produção usaria Redis ou banco):
  - Se taxa de falsos positivos recente para um tipo de specialty > 20%,
    aplica penalidade adicional de +5%.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO BASE
# ══════════════════════════════════════════════════════════════════════════

def _get_base_threshold() -> float:
    """Lê threshold base da variável de ambiente. Default: 0.70."""
    raw = os.getenv("BASE_THRESHOLD", "0.70")
    try:
        value = float(raw)
        if not 0.0 < value < 1.0:
            raise ValueError
        return value
    except ValueError:
        return 0.70


BASE_THRESHOLD = _get_base_threshold()


# ══════════════════════════════════════════════════════════════════════════
# FATORES DE AJUSTE POR ESPECIALIDADE
# ══════════════════════════════════════════════════════════════════════════

SPECIALTY_ADJUSTMENTS: dict[str, float] = {
    # Ajuste = fator multiplicativo sobre o BASE_THRESHOLD
    # > 1.0 → mais rigoroso (limiar mais alto)
    # < 1.0 → menos rigoroso (limiar mais baixo)
    "oncologia":        1.10,
    "psiquiatria":      1.05,
    "cardiologia":      1.05,
    "neurologia":       1.05,
    "pediatria":        1.05,
    "ginecologia":      1.00,
    "clinica_geral":    1.00,
    "medicina_familia": 1.00,
    "ortopedia":        1.00,
    "dermatologia":     0.95,  # documentos simples, fotográficos — menos rigor
    "nutricao":         0.95,  # receituários mais informais são aceitos
}


# ══════════════════════════════════════════════════════════════════════════
# HISTÓRICO DE ERROS (in-memory, janela deslizante de 100 eventos)
# ══════════════════════════════════════════════════════════════════════════

class ErrorHistory:
    """
    Rastreia taxa de falsos positivos por specialty.
    Em produção: substituir por Redis com TTL.

    Falso positivo aqui = documento aceito (is_valid=True) que foi
    posteriormente marcado como inválido por feedback humano.
    """

    _WINDOW_SIZE = 100
    _FP_PENALTY_THRESHOLD = 0.20   # > 20% de FP → penalidade

    def __init__(self):
        self._history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._WINDOW_SIZE)
        )

    def record(self, specialty: str, was_false_positive: bool) -> None:
        """Registra resultado de validação posterior."""
        self._history[specialty].append(1 if was_false_positive else 0)

    def fp_rate(self, specialty: str) -> float:
        """Taxa de falsos positivos recente para a specialty."""
        history = self._history[specialty]
        if not history:
            return 0.0
        return sum(history) / len(history)

    def needs_penalty(self, specialty: str) -> bool:
        return self.fp_rate(specialty) > self._FP_PENALTY_THRESHOLD


# Singleton (reset a cada restart da aplicação)
_error_history = ErrorHistory()


# ══════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def compute_adaptive_threshold(
    specialty: Optional[str] = None,
    extraction_method: str = "digital_pdf",
    word_count: int = 100,
) -> float:
    """
    Calcula o limiar adaptativo para a decisão de validação.

    Args:
        specialty:         especialidade clínica do contexto (opcional)
        extraction_method: como o texto foi extraído
        word_count:        número de palavras no texto extraído

    Returns:
        threshold ∈ [0.0, 1.0]
    """
    threshold = BASE_THRESHOLD

    # 1. Ajuste por especialidade
    if specialty:
        specialty_key = specialty.lower().replace(" ", "_").replace("-", "_")
        factor = SPECIALTY_ADJUSTMENTS.get(specialty_key, 1.00)
        threshold *= factor

    # 2. Ajuste por qualidade de extração
    if extraction_method == "ocr_scanned_pdf":
        threshold += 0.05  # OCR menos confiável → mais rigor
    elif extraction_method == "fallback_empty":
        threshold = 0.99   # Sem texto extraído → rejeitar quase sempre

    if word_count < 30:
        threshold += 0.10  # Texto muito curto → sinal de extração ruim

    # 3. Penalidade por histórico de erros
    if specialty and _error_history.needs_penalty(specialty):
        threshold += 0.05

    # Garantir limites [0, 1]
    return round(min(max(threshold, 0.0), 1.0), 4)


def record_feedback(specialty: str, was_false_positive: bool) -> None:
    """
    Endpoint de feedback para atualizar o histórico de erros.
    Seria chamado pela interface quando um especialista rejeita um doc aceito.
    """
    _error_history.record(specialty, was_false_positive)
