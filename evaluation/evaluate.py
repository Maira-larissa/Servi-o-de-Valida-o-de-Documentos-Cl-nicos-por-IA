"""
evaluate.py — Avaliação do classificador sobre o dataset de 30 documentos.

Uso:
    # Com a API rodando em localhost:8000
    python evaluation/evaluate.py

    # Ou apontar para outro host
    API_URL=http://localhost:8000 python evaluation/evaluate.py

Saída:
    - Tabela de resultados no terminal
    - evaluation/results.json (resultados completos)
    - evaluation/report.md (relatório em Markdown para o README)
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATASET_DIR = Path(__file__).parent / "dataset"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
REPORT_PATH = Path(__file__).parent / "report.md"


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def call_api(file_path: Path, specialty: str | None = None) -> dict[str, Any]:
    """Chama o endpoint /validate e retorna a resposta."""
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        data = {}
        if specialty:
            data["specialty"] = specialty

        resp = requests.post(f"{API_URL}/validate", files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()


def compute_metrics(results: list[dict]) -> dict[str, float]:
    """
    Calcula acurácia, precisão, recall e F1-score.

    Definição de positivo: is_valid=True (documento clínico aceito)
    """
    tp = fp = tn = fn = 0

    for r in results:
        gt    = r["ground_truth_is_valid"]
        pred  = r["prediction_is_valid"]

        if gt and pred:       tp += 1
        elif not gt and pred: fp += 1
        elif gt and not pred: fn += 1
        else:                 tn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
    }


# ══════════════════════════════════════════════════════════════════════════
# AVALIAÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation() -> None:
    print(f"Conectando à API: {API_URL}")
    print(f"Dataset: {DATASET_DIR}\n")

    # Verificar health
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        print(f"API OK — versão {health['version']} — threshold base: {health['threshold_base']}\n")
    except Exception as e:
        print(f"ERRO: API não disponível em {API_URL}: {e}")
        return

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    errors  = []

    for item in manifest:
        category = item["category"]
        filename = item["filename"]
        file_path = DATASET_DIR / category / filename

        if not file_path.exists():
            print(f"  AVISO: arquivo não encontrado: {file_path}")
            continue

        try:
            t0 = time.perf_counter()
            response = call_api(file_path)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 0)

            correct = (response["is_valid"] == item["ground_truth_is_valid"])

            result = {
                "filename":              filename,
                "category":              category,
                "ground_truth_label":    item["ground_truth_label"],
                "ground_truth_is_valid": item["ground_truth_is_valid"],
                "prediction_label":      response["label"],
                "prediction_is_valid":   response["is_valid"],
                "confidence_score":      response["confidence_score"],
                "threshold_applied":     response["threshold_applied"],
                "extraction_method":     response["extraction_method"],
                "clinical_terms_found":  response["clinical_terms_found"],
                "justification":         response["justification"],
                "correct":               correct,
                "elapsed_ms":            elapsed_ms,
                "notes":                 item.get("notes", ""),
            }
            results.append(result)

            status = "✓" if correct else "✗"
            print(
                f"  {status} [{category:10}] {filename:45} "
                f"| pred={str(response['is_valid']):5} gt={str(item['ground_truth_is_valid']):5} "
                f"| conf={response['confidence_score']:.2f} "
                f"| thr={response['threshold_applied']:.2f} "
                f"| {elapsed_ms:5.0f}ms"
            )

            if not correct:
                errors.append(result)

        except Exception as e:
            print(f"  ERRO: {filename}: {e}")

    # ── Métricas por categoria ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MÉTRICAS GLOBAIS")
    print("=" * 80)

    all_metrics   = compute_metrics(results)
    valid_metrics = compute_metrics([r for r in results if r["category"] == "valid"])
    inv_metrics   = compute_metrics([r for r in results if r["category"] == "invalid"])
    amb_metrics   = compute_metrics([r for r in results if r["category"] == "ambiguous"])

    print(f"\nGlobal ({all_metrics['total']} docs):")
    print(f"  Acurácia:  {all_metrics['accuracy']:.4f}")
    print(f"  Precisão:  {all_metrics['precision']:.4f}")
    print(f"  Recall:    {all_metrics['recall']:.4f}")
    print(f"  F1-Score:  {all_metrics['f1_score']:.4f}")
    print(f"  TP={all_metrics['tp']} FP={all_metrics['fp']} TN={all_metrics['tn']} FN={all_metrics['fn']}")

    print(f"\nVálidos ({valid_metrics['total']} docs):   F1={valid_metrics['f1_score']:.4f}")
    print(f"Inválidos ({inv_metrics['total']} docs): F1={inv_metrics['f1_score']:.4f}")
    print(f"Ambíguos ({amb_metrics['total']} docs):  F1={amb_metrics['f1_score']:.4f}")

    # ── Erros mais representativos ─────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"ERROS ({len(errors)} de {len(results)} documentos)")
    print("=" * 80)

    for err in errors:
        pred_str = "ACEITO" if err["prediction_is_valid"] else "REJEITADO"
        gt_str   = "deveria ACEITAR" if err["ground_truth_is_valid"] else "deveria REJEITAR"
        err_type = "FALSO POSITIVO" if (err["prediction_is_valid"] and not err["ground_truth_is_valid"]) else "FALSO NEGATIVO"

        print(f"\n  [{err_type}] {err['filename']}")
        print(f"    Predição: {pred_str} (conf={err['confidence_score']:.2f}, thr={err['threshold_applied']:.2f})")
        print(f"    Ground truth: {gt_str}")
        if err["notes"]:
            print(f"    Análise: {err['notes'][:200]}")

    # ── Salvar resultados ──────────────────────────────────────────────
    output = {
        "metrics": {
            "global":     all_metrics,
            "valid":      valid_metrics,
            "invalid":    inv_metrics,
            "ambiguous":  amb_metrics,
        },
        "results":         results,
        "errors_analysis": errors,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResultados salvos em: {RESULTS_PATH}")
    _generate_report(output)


# ══════════════════════════════════════════════════════════════════════════
# GERAÇÃO DO RELATÓRIO EM MARKDOWN
# ══════════════════════════════════════════════════════════════════════════

def _generate_report(data: dict) -> None:
    m = data["metrics"]["global"]
    errors = data["errors_analysis"]

    lines = [
        "# Relatório de Avaliação — Classificador de Documentos Clínicos",
        "",
        "## Tabela de Resultados Globais",
        "",
        "| Métrica   | Valor  |",
        "|-----------|--------|",
        f"| Acurácia  | {m['accuracy']:.4f} |",
        f"| Precisão  | {m['precision']:.4f} |",
        f"| Recall    | {m['recall']:.4f} |",
        f"| F1-Score  | {m['f1_score']:.4f} |",
        "",
        "### Matriz de Confusão",
        "",
        f"| | Predito: Válido | Predito: Inválido |",
        f"|---|---|---|",
        f"| **Real: Válido**   | TP={m['tp']} | FN={m['fn']} |",
        f"| **Real: Inválido** | FP={m['fp']} | TN={m['tn']} |",
        "",
        "## Erros Mais Representativos",
        "",
    ]

    for err in errors:
        err_type = (
            "🔴 Falso Positivo (doc inválido aceito)"
            if (err["prediction_is_valid"] and not err["ground_truth_is_valid"])
            else "🟡 Falso Negativo (doc válido rejeitado)"
        )
        lines += [
            f"### `{err['filename']}`",
            f"**Tipo de erro:** {err_type}  ",
            f"**Confiança:** {err['confidence_score']:.2f} | **Limiar:** {err['threshold_applied']:.2f}  ",
            f"**Análise:** {err.get('notes', 'Sem notas.')}",
            "",
        ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Relatório gerado em: {REPORT_PATH}")


if __name__ == "__main__":
    run_evaluation()
