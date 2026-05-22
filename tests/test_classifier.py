"""
Testes unitários do classificador.

Rodados com: pytest tests/ -v

Filosofia: documentar hipóteses de onde o modelo DEVE falhar antes
de executar (ML testing sistemático — bonus do edital).

Cada teste tem um comentário "Hipótese" explicando o que esperamos
e "Limitação conhecida" onde aplicável.
"""

import pytest
from app.classifier import ClinicalDocumentClassifier
from app.extractor import normalize_text
from app.threshold import compute_adaptive_threshold, BASE_THRESHOLD

clf = ClinicalDocumentClassifier()


# ══════════════════════════════════════════════════════════════════════════
# TESTES DO NORMALIZADOR
# ══════════════════════════════════════════════════════════════════════════

class TestNormalizer:
    def test_lowercase(self):
        assert normalize_text("LAUDO MÉDICO") == "laudo médico"

    def test_preserva_acentos(self):
        # Crítico: acentos diferenciam termos no vocabulário clínico
        result = normalize_text("Diagnóstico: Hipertensão")
        assert "diagnóstico" in result
        assert "hipertensão" in result

    def test_remove_pontuacao(self):
        result = normalize_text("CRM-PB: 12345, Dr. Silva!")
        assert "!" not in result
        assert "," not in result

    def test_colapsa_espacos(self):
        result = normalize_text("  paciente   com   febre  ")
        assert "  " not in result


# ══════════════════════════════════════════════════════════════════════════
# TESTES DO CLASSIFICADOR — CASOS CLAROS
# ══════════════════════════════════════════════════════════════════════════

class TestClassifierClearCases:

    def test_laudo_hemograma_aceito(self):
        """Hipótese: laudo com hemograma completo deve ter alta confiança."""
        text = normalize_text("""
            HEMOGRAMA COMPLETO
            Paciente: João Silva — Data: 15/05/2026
            Eritrócitos: 4,8 × 10⁶/µL  Leucócitos: 7.200 /µL
            Conclusão: Hemograma dentro dos parâmetros normais.
            Dr. Carlos Medeiros — CRM-PB 12345
        """)
        result = clf.classify(text, word_count=len(text.split()))
        assert result.confidence_score >= 0.60
        assert result.label in ("laudo", "relatorio_medico")

    def test_receita_medica_aceita(self):
        """Hipótese: receita com posologia e CRM deve ser aceita."""
        text = normalize_text("""
            RECEITA MÉDICA
            Losartana 50mg — via oral — 1 comprimido pela manhã
            Posologia: uso contínuo. Hipertensão arterial sistêmica CID I10.
            Dra. Ana Santos — CRM-PB 54321
        """)
        result = clf.classify(text, word_count=len(text.split()))
        assert result.confidence_score >= 0.55
        assert result.label in ("receita", "laudo", "relatorio_medico")

    def test_nota_fiscal_rejeitada(self):
        """Hipótese: nota fiscal com CNPJ e valor total deve ter score baixo."""
        text = normalize_text("""
            NOTA FISCAL ELETRÔNICA
            CNPJ: 11.222.333/0001-44
            Valor total: R$ 3.678,80
            Forma de pagamento: cartão de crédito em parcelas
            Vencimento: 15/06/2026
        """)
        result = clf.classify(text, word_count=len(text.split()))
        assert result.confidence_score <= 0.40
        assert result.label in ("fatura", "desconhecido")

    def test_contrato_rejeitado(self):
        """Hipótese: contrato com cláusulas deve ser rejeitado."""
        text = normalize_text("""
            CONTRATO DE LOCAÇÃO
            Cláusula 1ª: o locador cede o imóvel ao locatário.
            Cláusula 2ª: prazo de 30 meses a partir de 01/06/2026.
            Testemunhas: _____________  _____________
        """)
        result = clf.classify(text, word_count=len(text.split()))
        assert result.confidence_score <= 0.35
        assert result.label == "contrato"

    def test_texto_vazio_rejeitado(self):
        """Hipótese: texto vazio deve resultar em confiança mínima."""
        result = clf.classify("", word_count=0)
        assert result.confidence_score <= 0.15
        assert result.label == "selfie"


# ══════════════════════════════════════════════════════════════════════════
# TESTES — CASOS AMBÍGUOS (hipóteses de falha documentadas)
# ══════════════════════════════════════════════════════════════════════════

class TestAmbiguousCases:

    def test_artigo_jornalistico_saude(self):
        """
        Hipótese: NOSSO MODELO DEVE FALHAR AQUI.
        Um artigo jornalístico sobre saúde menciona termos clínicos
        (hipertensão, Losartana, cardiologista) mas NÃO é um documento clínico.
        Limitação conhecida: sem modelo de contexto semântico, não conseguimos
        distinguir 'texto sobre medicina' de 'documento médico'.
        Este é um falso positivo esperado.
        """
        text = normalize_text("""
            Segundo o Ministério da Saúde, 36 milhões de brasileiros têm hipertensão.
            O cardiologista explica que Losartana é amplamente usada no tratamento.
            O diagnóstico é feito pela medição da pressão arterial.
        """)
        result = clf.classify(text, word_count=len(text.split()))
        # NÃO fazemos assert de resultado correto — documentamos a limitação
        # O teste serve para monitorar se o comportamento muda
        print(f"\n[artigo_jornalistico] score={result.confidence_score:.2f} label={result.label}")
        # Apenas verificamos que retorna algo
        assert 0.0 <= result.confidence_score <= 1.0

    def test_exame_veterinario(self):
        """
        Hipótese: NOSSO MODELO DEVE FALHAR AQUI.
        Exame veterinário tem estrutura idêntica ao humano.
        Diferença: CRMV vs CRM. Nosso regex de registro profissional
        aceita ambos (padrão 'crm\s*\d+').
        """
        text = normalize_text("""
            CLÍNICA VETERINÁRIA — CRMV-PB 12345
            Paciente: Rex (cão, Labrador, 5 anos)
            Hemograma: Eritrócitos 6,2  Leucócitos 9.800  Plaquetas 320.000
            Conclusão: Hemograma dentro dos parâmetros normais para a espécie.
            Dra. Vanessa Melo — Médica Veterinária
        """)
        result = clf.classify(text, word_count=len(text.split()))
        print(f"\n[exame_veterinario] score={result.confidence_score:.2f} label={result.label}")
        # Documenta limitação, não força resultado
        assert 0.0 <= result.confidence_score <= 1.0

    def test_atestado_manuscrito_ocr_ruim(self):
        """
        Hipótese: atestado manuscrito com OCR degradado deve ser rejeitado
        ou ter confiança baixa. Risco de falso negativo importante
        (documento clínico legítimo rejeitado).
        """
        text = normalize_text(
            "atestado medico paciente jose silva repouso 2 dias dr antonio crm 78901"
        )
        result = clf.classify(text, word_count=len(text.split()))
        print(f"\n[atestado_manuscrito] score={result.confidence_score:.2f} label={result.label}")
        # Score pode ser baixo — documentamos que isso é um FN esperado
        assert 0.0 <= result.confidence_score <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# TESTES DO LIMIAR ADAPTATIVO
# ══════════════════════════════════════════════════════════════════════════

class TestAdaptiveThreshold:

    def test_base_threshold(self):
        """Limiar sem ajustes deve ser igual ao BASE_THRESHOLD."""
        t = compute_adaptive_threshold()
        assert t == pytest.approx(BASE_THRESHOLD, abs=0.001)

    def test_oncologia_mais_rigoroso(self):
        """Oncologia deve ter limiar maior que clínica geral."""
        t_onco    = compute_adaptive_threshold(specialty="oncologia")
        t_clinica = compute_adaptive_threshold(specialty="clinica_geral")
        assert t_onco > t_clinica

    def test_ocr_aumenta_threshold(self):
        """OCR em PDF escaneado deve aumentar o limiar."""
        t_digital = compute_adaptive_threshold(extraction_method="digital_pdf")
        t_ocr     = compute_adaptive_threshold(extraction_method="ocr_scanned_pdf")
        assert t_ocr > t_digital

    def test_texto_curto_aumenta_threshold(self):
        """Texto com < 30 palavras deve receber threshold maior."""
        t_curto   = compute_adaptive_threshold(word_count=10)
        t_longo   = compute_adaptive_threshold(word_count=200)
        assert t_curto > t_longo

    def test_threshold_nao_excede_1(self):
        """Threshold nunca deve ultrapassar 1.0."""
        t = compute_adaptive_threshold(
            specialty="oncologia",
            extraction_method="ocr_scanned_pdf",
            word_count=5,
        )
        assert t <= 1.0

    def test_fallback_empty_threshold_maximo(self):
        """Extração vazia deve resultar em threshold próximo de 1 (rejeitar quase sempre)."""
        t = compute_adaptive_threshold(extraction_method="fallback_empty")
        assert t >= 0.95


# ══════════════════════════════════════════════════════════════════════════
# TESTES DA JUSTIFICATIVA
# ══════════════════════════════════════════════════════════════════════════

class TestJustification:

    def test_justificativa_nao_vazia(self):
        """Justificativa deve sempre ter conteúdo."""
        text = normalize_text("Hemograma completo. Paciente João. CRM 12345.")
        result = clf.classify(text, word_count=len(text.split()))
        assert len(result.justification) > 10

    def test_justificativa_menciona_termos_clinicos(self):
        """Para doc clínico, justificativa deve mencionar termos encontrados."""
        text = normalize_text("Diagnóstico: hipertensão arterial. Prescrição: Losartana 50mg.")
        result = clf.classify(text, word_count=len(text.split()))
        if result.clinical_terms_found:
            assert any(t in result.justification for t in result.clinical_terms_found)
