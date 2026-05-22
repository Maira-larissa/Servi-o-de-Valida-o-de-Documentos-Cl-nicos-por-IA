"""
Contrato da API — definido ANTES da implementação.

Decisão de design: usar Pydantic v2 para validação automática.
Todos os campos foram pensados para que o módulo Fullstack (P01)
consiga consumir sem ambiguidade.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Tipos permitidos de rótulo ─────────────────────────────────────────────
VALID_LABELS = (
    "laudo",
    "receita",
    "atestado",
    "relatorio_medico",
    "exame_imagem",
    "selfie",
    "fatura",
    "contrato",
    "print_tela",
    "desconhecido",
)


class ClassificationRequest(BaseModel):
    """
    Parâmetros opcionais que acompanham o upload (via Form Data).
    O arquivo em si chega via multipart/form-data no endpoint.
    """
    specialty: Optional[str] = Field(
        default=None,
        description=(
            "Especialidade clínica do solicitante. "
            "Influencia o limiar adaptativo. "
            "Ex: 'oncologia', 'cardiologia', 'clinica_geral'."
        ),
        examples=["oncologia", "cardiologia", "psiquiatria"],
    )


class ClassificationResponse(BaseModel):
    """
    Resposta padronizada do classificador.

    Campos obrigatórios pelo edital:
      - is_valid         : decisão binária
      - confidence_score : probabilidade da classe predita (0–1)
      - label            : rótulo atribuído
      - justification    : texto legível explicando a decisão
      - threshold_applied: limiar efetivamente usado na decisão

    Campos extras (enriquecem a integração Fullstack):
      - clinical_terms_found : termos clínicos que influenciaram a decisão
      - extraction_method    : como o texto foi obtido (transparência)
      - word_count           : palavras extraídas (útil para debug)
    """

    is_valid: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    label: str
    justification: str
    threshold_applied: float = Field(..., ge=0.0, le=1.0)

    # Campos extras para explicabilidade e debug
    clinical_terms_found: list[str] = Field(default_factory=list)
    extraction_method: str = Field(
        description="'digital_pdf' | 'ocr_scanned_pdf' | 'ocr_image' | 'fallback_empty'"
    )
    word_count: int = Field(default=0)


class HealthResponse(BaseModel):
    status: str
    version: str
    threshold_base: float
