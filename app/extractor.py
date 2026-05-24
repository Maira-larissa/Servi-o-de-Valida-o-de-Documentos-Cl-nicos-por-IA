"""
Extração de texto de PDFs e imagens.
Implementação: RapidOCR (ONNX) - 100% Python, sem dependência de binários externos.
"""

import re
import io
from pathlib import Path
from typing import Tuple

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

# Singleton para otimizar performance: faz o modelo de IA ccarregar uma única vez
_ocr_engine = None

def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine

MIN_CHARS_PER_PAGE_TO_CONSIDER_DIGITAL = 50

def extract_from_pdf(file_bytes: bytes) -> Tuple[str, str]:
    if not _FITZ_AVAILABLE:
        return "", "fallback_empty"

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()

    avg_chars_per_page = len(raw_text.strip()) / max(doc.page_count, 1)

    if avg_chars_per_page < MIN_CHARS_PER_PAGE_TO_CONSIDER_DIGITAL:
        ocr_text = _extract_via_ocr_from_pdf(doc)
        doc.close()
        return normalize_text(ocr_text), "ocr_scanned_pdf"

    doc.close()
    return normalize_text(raw_text), "digital_pdf"

def extract_from_image(file_bytes: bytes) -> Tuple[str, str]:
    return normalize_text(_run_ocr_on_bytes(file_bytes)), "ocr_image"

def _extract_via_ocr_from_pdf(doc) -> str:
    engine = _get_ocr_engine()
    pages_text = []
    for page in doc:
        try:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            result, _ = engine(img_bytes)
            if result:
                pages_text.append("\n".join([res[1] for res in result]))
        except Exception:
            continue
    return "\n".join(pages_text)

def _run_ocr_on_bytes(file_bytes: bytes) -> str:
    try:
        engine = _get_ocr_engine()
        result, _ = engine(file_bytes)
        return "\n".join([res[1] for res in result]) if result else ""
    except Exception:
        return ""

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúâêîôûãõàèìòùç]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_file_type(filename: str, content_type: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" or content_type == "application/pdf":
        return "pdf"
    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp") or content_type.startswith("image/"):
        return "image"
    return "unknown"