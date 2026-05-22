from fastapi.concurrency import run_in_threadpool
import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.classifier import ClinicalDocumentClassifier
from app.extractor import detect_file_type, extract_from_image, extract_from_pdf
from app.schemas import ClassificationResponse, HealthResponse
from app.threshold import BASE_THRESHOLD, compute_adaptive_threshold, record_feedback

# ══════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Clinical Document Validator",
    description=(
        "Serviço de IA para validação de documentos clínicos. "
        "Parte do Projeto ReNTAI / CLUSTER-19 — LAVID/UFPB."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção: restringir ao domínio do Fullstack
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_classifier = ClinicalDocumentClassifier()


# ══════════════════════════════════════════════════════════════════════════
# LIMITES DE SEGURANÇA
# ══════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Infra"])
async def health_check() -> HealthResponse:
    """Verifica disponibilidade do serviço."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        threshold_base=BASE_THRESHOLD,
    )


@app.post(
    "/validate",
    response_model=ClassificationResponse,
    tags=["Classificação"],
    summary="Valida se um arquivo é um documento clínico legítimo",
    description="""
Recebe um arquivo (PDF ou imagem) e retorna a classificação.

**Campos de retorno:**
- `is_valid`: se o documento é aceito como clínico legítimo
- `confidence_score`: probabilidade (0–1) da classe predita
- `label`: rótulo atribuído (laudo, receita, fatura, etc.)
- `justification`: explicação legível da decisão
- `threshold_applied`: limiar efetivamente usado
- `clinical_terms_found`: termos clínicos que influenciaram o score
- `extraction_method`: como o texto foi obtido
    """,
)
async def validate_document(
    file: UploadFile = File(..., description="PDF ou imagem do documento"),
    specialty: Optional[str] = Form(
        default=None,
        description="Especialidade clínica (ex: oncologia, cardiologia)",
    ),
) -> ClassificationResponse:
    """Pipeline completo de validação de documento clínico."""

    t_start = time.perf_counter()

    # ── 1. Validações de entrada ───────────────────────────────────────

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Tipo de arquivo não suportado: {file.content_type}. "
                f"Aceito: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande ({size_mb:.1f} MB). Limite: {MAX_FILE_SIZE_MB} MB.",
        )

    # ── 2. Extração de texto ───────────────────────────────────────────
# ── 2. Extração de texto ───────────────────────────────────────────

    file_type = detect_file_type(file.filename or "", file.content_type or "")

    if file_type == "pdf":
        text, extraction_method = await run_in_threadpool(extract_from_pdf, file_bytes)
    elif file_type == "image":
        text, extraction_method = await run_in_threadpool(extract_from_image, file_bytes)
    else:
        text, extraction_method = "", "fallback_empty"

    word_count = len(text.split()) if text else 0

    # ── 3. Classificação ──────────────────────────────────────────────

    result = _classifier.classify(text, word_count)

    # ── 4. Limiar adaptativo ──────────────────────────────────────────

    threshold = compute_adaptive_threshold(
        specialty=specialty,
        extraction_method=extraction_method,
        word_count=word_count,
    )

    # ── 5. Decisão final ──────────────────────────────────────────────

    is_valid = (
        result.confidence_score >= threshold
        and result.label not in ("selfie", "fatura", "contrato", "print_tela")
    )

    # Enriquecer justificativa se rejeitado por threshold
    justification = result.justification
    if not is_valid and result.confidence_score < threshold:
        justification += (
            f" Confiança ({result.confidence_score:.2f}) abaixo do limiar "
            f"adaptativo ({threshold:.2f}) — documento rejeitado."
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return ClassificationResponse(
        is_valid=is_valid,
        confidence_score=result.confidence_score,
        label=result.label,
        justification=justification,
        threshold_applied=threshold,
        clinical_terms_found=result.clinical_terms_found,
        extraction_method=extraction_method,
        word_count=word_count,
    )


@app.post(
    "/feedback",
    tags=["Melhoria Contínua"],
    summary="Registra feedback de falso positivo para ajuste do limiar",
)
async def register_feedback(
    specialty: str = Form(...),
    was_false_positive: bool = Form(...),
):
    """
    Recebe feedback da interface (especialista marcou doc como inválido
    após ter sido aceito). Ajusta o histórico de erros para a specialty.
    """
    record_feedback(specialty, was_false_positive)
    return {"status": "feedback registrado", "specialty": specialty}
