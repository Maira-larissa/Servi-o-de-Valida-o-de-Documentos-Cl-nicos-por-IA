"""
Classificador de Documentos Clínicos.

─── DECISÃO DE ABORDAGEM ───────────────────────────────────────────────────

Abordagem escolhida: Classificador híbrido baseado em vocabulário clínico
ponderado + heurísticas estruturais.

Por que NÃO foi escolhido um modelo de linguagem treinável (DistilBERT, TF-IDF
+ LogReg):
  - Sem corpus rotulado real disponível no prazo do desafio.
  - Treinar com dados sintéticos cria viés artificial: o modelo aprenderia os
    próprios exemplos, não a tarefa real.
  - Resultado: F1 inflado no conjunto sintético, desempenho imprevisível em
    dados reais de teleconsultoria.

Por que a abordagem de vocabulário é honesta e adequada aqui:
  - O vocabulário médico em português é suficientemente distinto do não-médico.
  - A decisão é 100% auditável: sabe-se exatamente quais termos levaram ao score.
  - Não há "caixa preta" — o edital valoriza explicabilidade.
  - Limitação reconhecida: vulnerável a documentos que citam termos médicos sem
    ser clínicos (ex: artigo jornalístico sobre saúde). Documentado na análise
    crítica.

─── ESTRUTURA DO SCORE ─────────────────────────────────────────────────────

  score_clinico  = soma ponderada de termos clínicos encontrados (normalizada)
  score_estrutural = presença de elementos estruturais (CRM, data, assinatura etc.)

  Penalidade: se termos negativos (fatura, contrato etc.) são dominantes,
  o score é penalizado diretamente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════════════
# VOCABULÁRIO CLÍNICO PONDERADO
# Peso 3 → terminologia altamente específica do contexto médico
# Peso 2 → termos frequentes em documentos clínicos mas não exclusivos
# Peso 1 → termos de suporte / contexto
# ══════════════════════════════════════════════════════════════════════════

CLINICAL_VOCABULARY: Dict[str, int] = {
    # ── Peso 3: Alta especificidade clínica ───────────────────────────
    "laudo": 3, "diagnóstico": 3, "diagnose": 3,
    "hipótese diagnóstica": 3, "cid": 3, "cid-10": 3,
    "prescrição": 3, "prescrevo": 3, "prescrito": 3,
    "posologia": 3, "dose": 3, "via oral": 3, "sublingual": 3,
    "hemograma": 3, "leucócitos": 3, "eritrócitos": 3, "plaquetas": 3,
    "tomografia": 3, "ressonância magnética": 3, "ultrassonografia": 3,
    "eletrocardiograma": 3, "ecocardiograma": 3,
    "anamnese": 3, "prontuário": 3, "evolução clínica": 3,
    "glicemia": 3, "colesterol": 3, "triglicerídeos": 3,
    "creatinina": 3, "ureia": 3, "tsh": 3, "t4 livre": 3,
    "pressão arterial": 3, "pa": 2,
    "hipertensão": 3, "diabetes": 3, "hipotensão": 3,
    "infarto": 3, "isquemia": 3, "neoplasia": 3, "tumor": 2,
    "fratura": 3, "luxação": 3, "edema": 3,
    "receita médica": 3, "receituário": 3,
    "atestado médico": 3, "afastamento": 2,
    "crm": 3, "crf": 3, "coren": 3,
    "exame laboratorial": 3, "exame de imagem": 3,

    # ── Peso 2: Termos frequentes em documentos clínicos ──────────────
    "paciente": 2, "enfermeiro": 2, "médico": 2, "doutor": 2, "dra": 2,
    "hospital": 2, "clínica": 2, "ubs": 2, "upa": 2, "sus": 2,
    "consulta": 2, "retorno": 2, "encaminhamento": 2, "referência": 2,
    "sintoma": 2, "queixa": 2, "dor": 2, "febre": 2, "náusea": 2,
    "tratamento": 2, "medicamento": 2, "remédio": 2, "comprimido": 2,
    "mg": 2, "ml": 2, "ampola": 2, "cápsula": 2,
    "exame": 2, "resultado": 2, "análise": 2, "amostra": 2,
    "sangue": 2, "urina": 2, "fezes": 2,
    "radiografia": 2, "rx": 2, "raio-x": 2, "raio x": 2,
    "cirurgia": 2, "procedimento": 2, "internação": 2, "alta": 2,
    "relatório médico": 2, "relatório clínico": 2,

    # ── Peso 1: Termos de suporte / contexto ──────────────────────────
    "saúde": 1, "doença": 1, "patologia": 1, "síndrome": 1,
    "infecção": 1, "inflamação": 1, "crônico": 1, "agudo": 1,
    "ambulatório": 1, "pronto-socorro": 1, "emergência": 1,
    "nome": 1, "data": 1, "assinatura": 1,
}

# ══════════════════════════════════════════════════════════════════════════
# VOCABULÁRIO NEGATIVO (forte indicador de documento NÃO clínico)
# ══════════════════════════════════════════════════════════════════════════

NON_CLINICAL_VOCABULARY: Dict[str, int] = {
    # Documentos financeiros
    "nota fiscal": 4, "fatura": 4, "boleto": 4, "nf-e": 4,
    "cnpj": 3, "valor total": 3, "subtotal": 3, "desconto": 2,
    "parcela": 2, "juros": 2, "vencimento": 2,
    "pagamento": 2, "cobrança": 2, "banco": 2,

    # Documentos jurídicos / contratuais
    "contrato": 4, "cláusula": 4, "rescisão": 3, "locação": 3,
    "arrendamento": 3, "parte contratante": 4, "parte contratada": 4,
    "testemunha": 2, "cartório": 2, "reconhecimento de firma": 3,

    # Prints / capturas de tela
    "screenshot": 4, "print screen": 4, "captura de tela": 3,
    "whatsapp": 3, "instagram": 3, "facebook": 3, "twitter": 3,

    # Selfie / foto pessoal (texto raramente presente, mas OCR pode pegar)
    "selfie": 4,

    # Publicidade
    "promoção": 2, "desconto especial": 3, "oferta": 2,
}

# ══════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE RÓTULOS
# ══════════════════════════════════════════════════════════════════════════

LABEL_PATTERNS: Dict[str, List[str]] = {
    "laudo":           ["laudo", "resultado", "exame laboratorial", "hemograma",
                        "tomografia", "ressonância", "eletrocardiograma"],
    "receita":         ["receita médica", "prescrição", "prescrevo", "posologia",
                        "uso contínuo", "via oral"],
    "atestado":        ["atestado médico", "afastamento", "apto", "inapto"],
    "relatorio_medico":["relatório médico", "relatório clínico", "evolução clínica",
                        "anamnese", "prontuário"],
    "exame_imagem":    ["tomografia", "ressonância magnética", "ultrassonografia",
                        "radiografia", "raio-x"],
    "selfie":          [],   # detectado pela ausência de termos clínicos + texto mínimo
    "fatura":          ["nota fiscal", "fatura", "boleto", "cnpj"],
    "contrato":        ["contrato", "cláusula", "parte contratante"],
    "print_tela":      ["screenshot", "print screen", "captura de tela", "whatsapp"],
}


# ══════════════════════════════════════════════════════════════════════════
# DATACLASS DE RESULTADO INTERNO
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ClassificationResult:
    label: str
    confidence_score: float
    clinical_terms_found: List[str] = field(default_factory=list)
    justification: str = ""


# ══════════════════════════════════════════════════════════════════════════
# CLASSIFICADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

class ClinicalDocumentClassifier:
    """
    Classificador baseado em vocabulário ponderado.

    Fluxo:
      1. Calcula score_clinico  (termos positivos)
      2. Calcula score_negativo (termos não clínicos)
      3. Calcula score_estrutural (CRM, CRF, padrão de data, etc.)
      4. Combina em score_final
      5. Determina rótulo predominante
      6. Gera justificativa legível
    """

    # Pesos da fusão
    _W_CLINICAL    = 0.70
    _W_STRUCTURAL  = 0.30
    _W_NEGATIVE    = 0.30   # peso da penalidade

    # Normalização: score_clinico_raw é dividido por este valor
    # para mapear para [0, 1]. Ajustado empiricamente.
    _CLINICAL_NORM = 12.0

    def classify(self, text: str, word_count: int) -> ClassificationResult:
        """
        Classifica um texto normalizado.

        Args:
            text: texto já normalizado (lowercase, sem caracteres especiais)
            word_count: número de palavras (usado para penalizar textos muito curtos)
        """
        if word_count < 5:
            return ClassificationResult(
                label="selfie",
                confidence_score=0.1,
                justification="Texto extraído insuficiente para análise. "
                              "Provável imagem sem texto (selfie ou foto genérica).",
            )

        clinical_score_raw, found_terms = self._score_clinical_terms(text)
        negative_score_raw = self._score_negative_terms(text)
        structural_score = self._score_structural_features(text)

        # Normalizar scores para [0, 1]
        clinical_score = min(clinical_score_raw / self._CLINICAL_NORM, 1.0)
        penalty        = min(negative_score_raw / 8.0, 1.0)

        # Fusion
        score_before_penalty = (
            self._W_CLINICAL   * clinical_score +
            self._W_STRUCTURAL * structural_score
        )
        # Penalidade subtrai proporcionalmente
        final_score = max(0.0, score_before_penalty - self._W_NEGATIVE * penalty)

        # Rótulo
        label = self._determine_label(text, clinical_score_raw, negative_score_raw, word_count)

        # Justificativa
        justification = self._build_justification(
            label, found_terms, clinical_score, structural_score, penalty, final_score
        )

        return ClassificationResult(
            label=label,
            confidence_score=round(final_score, 4),
            clinical_terms_found=found_terms[:10],  # top 10 termos
            justification=justification,
        )

    # ── Scoring de termos clínicos ─────────────────────────────────────

    def _score_clinical_terms(self, text: str) -> Tuple[float, List[str]]:
        total_weight = 0.0
        found = []

        for term, weight in CLINICAL_VOCABULARY.items():
            # Usar word boundary para evitar falsos matches (ex: "pa" em "pasta")
            pattern = r"\b" + re.escape(term) + r"\b"
            matches = re.findall(pattern, text)
            if matches:
                total_weight += weight * len(matches)
                found.append(term)

        return total_weight, found

    def _score_negative_terms(self, text: str) -> float:
        total = 0.0
        for term, weight in NON_CLINICAL_VOCABULARY.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text):
                total += weight
        return total

    # ── Scoring estrutural ─────────────────────────────────────────────

    def _score_structural_features(self, text: str) -> float:
        """
        Verifica características estruturais de documentos clínicos.
        Cada feature contribui igualmente (normalizado para 0-1).
        """
        features = {
            "registro_profissional": bool(re.search(
                r"\b(crm|crf|coren)\s*[:\-]?\s*\d{4,8}", text
            )),
            "data_formatada": bool(re.search(
                r"\b(crm|crf|coren)[a-z\-\/]*\s*[:\-]?\s*\d{4,8}", text
            )),
            "codigo_cid": bool(re.search(
                r"\bcid\b.{0,20}[a-z]\d{2}", text
            )),
            "codigo_tuss": bool(re.search(
                r"\btuss\b|\bprocedimento\b.{0,30}\d{5,}", text
            )),
            "assinatura_pattern": bool(re.search(
                r"(assinatura|ass\.|rubrica|carimbo)", text
            )),
            "paciente_nome": bool(re.search(
                r"\b(paciente|nome do paciente)\b.{0,50}[a-záéíóú]+", text
            )),
        }

        score = sum(features.values()) / len(features)
        return score

    # ── Determinação de rótulo ─────────────────────────────────────────

    def _determine_label(
        self,
        text: str,
        clinical_raw: float,
        negative_raw: float,
        word_count: int,
    ) -> str:
        # Dominância negativa → documento não clínico
        if negative_raw > clinical_raw and negative_raw >= 4:
            for label, patterns in [
                ("fatura",   ["nota fiscal", "fatura", "boleto", "cnpj"]),
                ("contrato", ["contrato", "cláusula"]),
                ("print_tela", ["screenshot", "whatsapp"]),
            ]:
                if any(p in text for p in patterns):
                    return label
            return "desconhecido"

        # Texto muito curto e sem termos → selfie
        if word_count < 20 and clinical_raw < 2:
            return "selfie"

        # Detectar rótulo pelo padrão dominante
        label_scores: Dict[str, int] = {}
        for label, patterns in LABEL_PATTERNS.items():
            label_scores[label] = sum(1 for p in patterns if p in text)

        best_label = max(label_scores, key=lambda k: label_scores[k])

        # Se nenhum padrão forte → label genérico clínico ou desconhecido
        if label_scores[best_label] == 0:
            return "desconhecido" if clinical_raw < 3 else "relatorio_medico"

        return best_label

    # ── Geração de justificativa ───────────────────────────────────────

    def _build_justification(
        self,
        label: str,
        found_terms: List[str],
        clinical_score: float,
        structural_score: float,
        penalty: float,
        final_score: float,
    ) -> str:
        parts = []

        if found_terms:
            terms_str = ", ".join(f"'{t}'" for t in found_terms[:5])
            parts.append(f"Termos clínicos identificados: {terms_str}.")

        if structural_score > 0.3:
            parts.append(
                f"Elementos estruturais de documento médico presentes "
                f"(score estrutural: {structural_score:.2f})."
            )

        if penalty > 0.2:
            parts.append(
                f"Presença de termos não clínicos (fatura, contrato, etc.) "
                f"aplicou penalidade de {penalty:.2f}."
            )

        if not parts:
            parts.append("Nenhum indicador clínico significativo encontrado.")

        label_map = {
            "laudo":            "Documento classificado como laudo/resultado de exame.",
            "receita":          "Documento classificado como receita/prescrição médica.",
            "atestado":         "Documento classificado como atestado médico.",
            "relatorio_medico": "Documento classificado como relatório clínico.",
            "exame_imagem":     "Documento classificado como exame de imagem.",
            "selfie":           "Documento classificado como imagem não clínica (selfie/foto).",
            "fatura":           "Documento classificado como fatura/nota fiscal.",
            "contrato":         "Documento classificado como contrato.",
            "print_tela":       "Documento classificado como print de tela.",
            "desconhecido":     "Tipo de documento não identificado com certeza.",
        }
        parts.insert(0, label_map.get(label, ""))

        return " ".join(filter(None, parts))
