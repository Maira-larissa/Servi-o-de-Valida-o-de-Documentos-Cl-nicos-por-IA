"""
Gerador de Dataset de Avaliação — 30 documentos sintéticos.

Estrutura:
  evaluation/dataset/
    ├── valid/      (10 documentos clínicos válidos)
    ├── invalid/    (10 documentos não clínicos)
    └── ambiguous/  (10 casos de borda)

Critério de seleção dos casos de borda (ambíguos):
  - Atestado médico manuscrito escaneado (OCR ruim)
  - Foto de monitor exibindo um exame (double-compression)
  - Artigo jornalístico sobre saúde (menciona termos médicos, não é doc clínico)
  - Receita de farmácia sem CRM legível
  - Exame veterinário (estrutura similar ao humano)
  - Declaração de comparecimento (formato de atestado, sem diagnóstico)
  - Print de app de telemedicina (mistura de UI + conteúdo clínico)
  - Relatório nutricional (limiar entre clínico e não clínico)
  - Laudo de exame de outra língua (espanhol — OCR pode confundir)
  - PDF corrompido com texto parcial

"""

import json
import os
from pathlib import Path

from fpdf import FPDF  # pip install fpdf2


OUTPUT_DIR = Path(__file__).parent / "dataset"


# ══════════════════════════════════════════════════════════════════════════
# TEXTOS DOS DOCUMENTOS
# ══════════════════════════════════════════════════════════════════════════

VALID_DOCS = [
    {
        "filename": "laudo_hemograma_01.pdf",
        "label": "laudo",
        "text": """
LABORATÓRIO CENTRAL - ANÁLISES CLÍNICAS
CNPJ: 00.000.000/0001-00  CRF: 12345-PB

Paciente: João da Silva
Data de nascimento: 01/01/1980
Data do exame: 15/05/2026

HEMOGRAMA COMPLETO

Eritrócitos: 4,8 × 10⁶/µL   (Ref: 4,5–5,5)
Hemoglobina: 14,2 g/dL       (Ref: 13,0–17,0)
Leucócitos:  7.200 /µL       (Ref: 4.000–10.000)
Plaquetas:   210.000 /µL     (Ref: 150.000–400.000)

Conclusão: Hemograma dentro dos parâmetros normais.

Dr. Carlos Medeiros — CRM-PB 12345
Assinatura: ___________________
        """,
    },
    {
        "filename": "receita_medica_01.pdf",
        "label": "receita",
        "text": """
RECEITA MÉDICA

Paciente: Maria Oliveira
Data: 12/05/2026

1. Losartana 50mg
   Posologia: 1 comprimido via oral pela manhã
   Uso contínuo

2. Atorvastatina 20mg
   Posologia: 1 comprimido via oral à noite
   Uso contínuo

Diagnóstico: Hipertensão arterial sistêmica (CID I10)

Dra. Ana Paula Santos
CRM-PB 54321
Carimbo e assinatura
        """,
    },
    {
        "filename": "atestado_medico_01.pdf",
        "label": "atestado",
        "text": """
ATESTADO MÉDICO

Atesto para os devidos fins que o(a) paciente PEDRO ALVES,
portador(a) do CPF 000.000.000-00, esteve sob minha avaliação clínica
em 14/05/2026, sendo recomendado repouso de 3 (três) dias.

CID-10: J06.9 — Infecção aguda das vias aéreas superiores não especificada.

Sem maiores informações no momento,

Dr. Roberto Lima — CRM-PB 67890
João Pessoa, 14 de maio de 2026
        """,
    },
    {
        "filename": "laudo_tomografia_01.pdf",
        "label": "exame_imagem",
        "text": """
CLÍNICA DE DIAGNÓSTICO POR IMAGEM

Exame: Tomografia Computadorizada de Crânio sem contraste
Paciente: Fernanda Costa   Data: 10/05/2026

ACHADOS:
Parênquima cerebral com densidade e morfologia preservadas.
Sistema ventricular com calibre normal. Não evidenciados sinais de
lesão expansiva, isquemia ou hemorragia intracraniana.

CONCLUSÃO: Tomografia de crânio sem alterações significativas.

Dr. Marcos Vinicius Almeida — CRM-PB 11111
Médico Radiologista
        """,
    },
    {
        "filename": "relatorio_medico_01.pdf",
        "label": "relatorio_medico",
        "text": """
RELATÓRIO MÉDICO — ENCAMINHAMENTO

Encaminho para avaliação especializada o(a) paciente LUCIA SANTOS,
50 anos, portadora de diabetes mellitus tipo 2 (CID E11) e hipertensão
arterial sistêmica (CID I10) em acompanhamento nesta UBS há 3 anos.

Evolução clínica: controle glicêmico irregular (HbA1c 9,2% em março/2026).
Está em uso de Metformina 850mg 2x/dia e Losartana 50mg/dia.

Solicito avaliação por endocrinologista para ajuste terapêutico.

Dra. Beatriz Nunes — CRM-PB 22222
UBS Centro — João Pessoa/PB
Data: 16/05/2026
        """,
    },
    {
        "filename": "laudo_ecg_01.pdf",
        "label": "laudo",
        "text": """
ELETROCARDIOGRAMA — ECG DE REPOUSO

Paciente: Hugo Ferreira   Idade: 65 anos   Data: 13/05/2026

Frequência cardíaca: 72 bpm
Ritmo: Sinusal regular
Eixo elétrico: Normal (60°)
Intervalo PR: 0,16s   QRS: 0,08s   QTc: 0,42s

Conclusão: Eletrocardiograma dentro dos limites da normalidade para a
faixa etária. Ausência de alterações isquêmicas ou de repolarização.

Dr. Sandro Melo — CRM-PB 33333 — Cardiologista
        """,
    },
    {
        "filename": "resultado_glicemia_01.pdf",
        "label": "laudo",
        "text": """
RESULTADO DE EXAME LABORATORIAL

Exame: Glicemia de jejum
Paciente: Carla Mendes   Data coleta: 08/05/2026

Glicemia: 98 mg/dL   Referência: 70–99 mg/dL (normal)

Colesterol Total: 195 mg/dL   Ref: < 200 mg/dL
HDL: 52 mg/dL          LDL: 128 mg/dL   Triglicerídeos: 75 mg/dL

Creatinina: 0,9 mg/dL   Ureia: 28 mg/dL   TSH: 2,1 µUI/mL

Responsável técnico: Farmacêutica Dra. Rita Fontes — CRF-PB 9876
        """,
    },
    {
        "filename": "prescricao_psiquiatria_01.pdf",
        "label": "receita",
        "text": """
PRESCRIÇÃO MÉDICA — CONTROLE ESPECIAL (Portaria 344/98)

Paciente: Bruno Alves   Data: 15/05/2026
CID-10: F32.1 — Episódio depressivo moderado

1. Sertralina 50mg — via oral — 1 comprimido pela manhã (30 dias)
2. Alprazolam 0,25mg — via oral — 1 comprimido à noite se necessário (15 dias)

Dr. Henrique Vasconcelos — CRM-PB 44444 — Psiquiatra
Receituário B2 nº 000001
        """,
    },
    {
        "filename": "laudo_ultrassom_01.pdf",
        "label": "exame_imagem",
        "text": """
ULTRASSONOGRAFIA DE ABDOME TOTAL

Paciente: Juliana Torres   Data: 11/05/2026

FÍGADO: Dimensões normais, ecotextura homogênea. Sem nódulos.
VESÍCULA: Paredes finas, sem litíase. Ausência de edema pericolecístico.
PÂNCREAS: Cabeça, corpo e cauda visíveis. Sem dilatação ductal.
RINS: Dimensões normais bilateralmente. Cortical preservada.

Conclusão: Abdome superior sem alterações ultrassonográficas.

Dr. Paulo Ribeiro — CRM-PB 55555 — Radiologista
Data do laudo: 11/05/2026
        """,
    },
    {
        "filename": "encaminhamento_aps_01.pdf",
        "label": "relatorio_medico",
        "text": """
GUIA DE ENCAMINHAMENTO — SUS

Unidade de Saúde: UBS Bairro Novo — João Pessoa/PB
Data: 16/05/2026

Paciente: Tatiana Lima   Idade: 42 anos
Hipótese diagnóstica: Neoplasia de colo uterino em investigação (CID C53)

Solicita-se consulta em oncologia de referência com urgência.
Paciente realizou colposcopia em 20/04/2026 com resultado sugestivo
de lesão de alto grau.

Dra. Sandra Alves — CRM-PB 66666
        """,
    },
]

INVALID_DOCS = [
    {
        "filename": "nota_fiscal_01.pdf",
        "label": "fatura",
        "text": """
NOTA FISCAL ELETRÔNICA — NF-e
CNPJ: 11.222.333/0001-44
Razão Social: Comércio de Eletrônicos Ltda.

Descrição dos produtos:
- Notebook Dell Inspiron 15   Qtd: 1   Valor: R$ 3.499,00
- Mouse sem fio               Qtd: 2   Valor: R$ 89,90

Subtotal: R$ 3.678,80
Desconto: R$ 0,00
TOTAL: R$ 3.678,80

Forma de pagamento: Cartão de crédito em 10 parcelas de R$ 367,88
Vencimento: 15/06/2026
        """,
    },
    {
        "filename": "contrato_locacao_01.pdf",
        "label": "contrato",
        "text": """
CONTRATO DE LOCAÇÃO RESIDENCIAL

Pelo presente instrumento particular, as partes abaixo identificadas:

PARTE LOCADORA: José Carlos Souza, CPF 111.222.333-44
PARTE LOCATÁRIA: Amanda Freitas, CPF 555.666.777-88

Cláusula 1ª: O locador cede ao locatário, para uso residencial,
o imóvel situado à Rua das Flores, 100 — João Pessoa/PB.

Cláusula 2ª: O prazo de locação é de 30 meses, iniciando em 01/06/2026.
Cláusula 3ª: O valor do aluguel mensal é de R$ 1.200,00, com reajuste anual.

Testemunhas: 1. _____________  2. _____________
        """,
    },
    {
        "filename": "fatura_energia_01.pdf",
        "label": "fatura",
        "text": """
ENERGISA — FATURA DE ENERGIA ELÉTRICA

Cliente: Marcos Pereira
Endereço: Av. Epitácio Pessoa, 500 — João Pessoa/PB
Mês referência: Abril/2026

Consumo: 320 kWh
Valor da energia: R$ 198,40
Iluminação pública: R$ 22,50
Tributos (ICMS + PIS/COFINS): R$ 67,20

TOTAL A PAGAR: R$ 288,10
Vencimento: 25/05/2026
Código de barras: 00000.00000 00000.000000 00000.000000 0 00000000000000
        """,
    },
    {
        "filename": "boleto_bancario_01.pdf",
        "label": "fatura",
        "text": """
BOLETO BANCÁRIO — BRADESCO

Cedente: Escola de Cursos Online LTDA
CNPJ: 22.333.444/0001-55

Sacado: Rodrigo Almeida
Valor do documento: R$ 349,00
Data de vencimento: 20/05/2026
Nosso número: 1234567890

Banco do Brasil — Agência 1234-5   Conta: 00001-2
Pagável em qualquer banco até o vencimento.
Após vencimento, consulte o banco emissor.
        """,
    },
    {
        "filename": "print_whatsapp_01.pdf",
        "label": "print_tela",
        "text": """
[Screenshot de conversa no WhatsApp]

Maria: Oi! Como você está?
Pedro: Bem! Fui ao médico hoje
Maria: E aí? Tudo certo?
Pedro: Sim, só uma gripe. Me receitou repouso
Maria: Que bom! Se cuida
Pedro: Valeu! Até amanhã
[Captura de tela - 14/05/2026 20:15]
        """,
    },
    {
        "filename": "curriculo_01.pdf",
        "label": "desconhecido",
        "text": """
CURRÍCULO VITAE

Nome: Leticia Andrade
E-mail: leticia@email.com   Telefone: (83) 99999-0000

FORMAÇÃO ACADÊMICA
Graduação em Administração — UFPB (2018–2022)
MBA em Gestão de Projetos — FGV (2023–2024)

EXPERIÊNCIA PROFISSIONAL
Analista de RH — Empresa XYZ (2022–2024)
Coordenadora de Projetos — Empresa ABC (2024–atual)

HABILIDADES
Excel avançado, Power BI, Scrum, liderança de equipes.
        """,
    },
    {
        "filename": "cardapio_restaurante_01.pdf",
        "label": "desconhecido",
        "text": """
RESTAURANTE BOM SABOR — CARDÁPIO DO DIA

Prato do dia: Frango grelhado com arroz, feijão e salada — R$ 18,00
Prato vegetariano: Quibe de grão-de-bico com tabule — R$ 16,00
Sopa do dia: Caldo verde — R$ 9,00

Bebidas:
Suco natural (500ml): R$ 7,00
Refrigerante lata: R$ 5,00
Água mineral: R$ 3,00

Sobremesas:
Pudim de leite condensado: R$ 6,00
        """,
    },
    {
        "filename": "proposta_comercial_01.pdf",
        "label": "contrato",
        "text": """
PROPOSTA COMERCIAL

De: TechSolutions Informática
Para: Prefeitura Municipal de João Pessoa

Apresentamos nossa proposta para fornecimento de sistema de gestão:

Módulo 1: Controle de estoque — R$ 15.000,00
Módulo 2: Relatórios gerenciais — R$ 8.000,00
Módulo 3: Suporte anual — R$ 5.000,00/ano

Validade da proposta: 30 dias
Prazo de entrega: 90 dias após assinatura do contrato
        """,
    },
    {
        "filename": "comprovante_residencia_01.pdf",
        "label": "desconhecido",
        "text": """
DECLARAÇÃO DE RESIDÊNCIA

Eu, FRANCISCO SANTOS, CPF 999.888.777-66, declaro para os devidos
fins que resido no endereço Rua do Sol, 250, Bairro Centro,
João Pessoa — PB, CEP 58000-000, desde janeiro de 2020.

Por ser verdade, assino o presente documento.

João Pessoa, 16 de maio de 2026.

___________________________
Francisco Santos
        """,
    },
    {
        "filename": "manual_produto_01.pdf",
        "label": "desconhecido",
        "text": """
MANUAL DO USUÁRIO — SMART TV 55"

Parabéns pela sua compra!

INSTALAÇÃO RÁPIDA:
1. Conecte o cabo de energia
2. Pressione o botão Liga/Desliga
3. Siga o assistente de configuração na tela
4. Conecte ao Wi-Fi de sua preferência

CONTROLE REMOTO:
- Botão HOME: menu principal
- Botão BACK: voltar
- Botão SOURCE: trocar entrada

Para suporte: 0800-000-0000 (seg–sex, 8h–18h)
        """,
    },
]

AMBIGUOUS_DOCS = [
    {
        "filename": "ambig_artigo_saude_01.pdf",
        "label": "desconhecido",  # Ground truth: inválido
        "text": """
JORNAL DA SAÚDE — EDIÇÃO MAIO 2026

"Hipertensão: o inimigo silencioso"

Segundo o Ministério da Saúde, cerca de 36 milhões de brasileiros
sofrem de hipertensão arterial. A doença, que raramente apresenta
sintomas, é um dos principais fatores de risco para infarto e AVC.

O cardiologista Dr. Paulo Saúde (nome fictício) explica:
"O diagnóstico é simples, feito com medição da pressão arterial.
Medicamentos como Losartana e Hidroclorotiazida são amplamente
usados no tratamento."

Dica: Meça sua pressão regularmente e mantenha consultas periódicas.
        """,
        "notes": (
            "Caso ambíguo: menciona termos clínicos (hipertensão, infarto, "
            "Losartana, cardiologista, diagnóstico) mas é texto jornalístico. "
            "Nosso classificador tende a aceitar erroneamente. "
            "Limitação conhecida: sem modelo de contexto, não distingue "
            "'texto sobre medicina' de 'documento médico'."
        ),
    },
    {
        "filename": "ambig_atestado_manuscrito_01.pdf",
        "label": "atestado",  # Ground truth: válido
        "text": """
atestado medico

atesto que o paciente jose silva esteve sob
minha avaliacao em 14 05 2026 recomendando
repouso por 2 dias

dr antonio crm pb 78901
        """,
        "notes": (
            "OCR de atestado manuscrito degrada a qualidade do texto. "
            "Sem acentos, sem pontuação, CRM parcialmente ilegível. "
            "Nosso classificador pode rejeitar por score baixo."
        ),
    },
    {
        "filename": "ambig_exame_veterinario_01.pdf",
        "label": "desconhecido",  # Ground truth: inválido (veterinário, não humano)
        "text": """
CLÍNICA VETERINÁRIA BICHO FELIZ
CRMV-PB: 12345

Paciente: Rex (cão, Labrador, 5 anos)
Tutor: Carlos Rodrigues

Exame: Hemograma completo
Data: 10/05/2026

Eritrócitos: 6,2 × 10⁶/µL   Hemoglobina: 15,1 g/dL
Leucócitos: 9.800 /µL        Plaquetas: 320.000 /µL

Conclusão: Hemograma dentro dos parâmetros normais para a espécie.

Dra. Vanessa Melo — CRMV-PB 12345 — Médica Veterinária
        """,
        "notes": (
            "Exame veterinário tem estrutura idêntica ao humano: "
            "hemograma, leucócitos, eritrócitos, assinatura com registro profissional. "
            "Diferença: CRMV (veterinário) vs CRM (humano). "
            "Nosso classificador não distingue — limitação séria em contexto de saúde humana."
        ),
    },
    {
        "filename": "ambig_declaracao_comparecimento_01.pdf",
        "label": "desconhecido",  # Ground truth: ambíguo (válido para presença, não clínico)
        "text": """
DECLARAÇÃO DE COMPARECIMENTO

Declaramos que o(a) Sr(a). PATRÍCIA MOURA compareceu a esta unidade
de saúde (UBS Central — João Pessoa/PB) no dia 15/05/2026 no horário
das 09h00 às 10h30.

Atenciosamente,

Recepção da UBS Central
Carimbo da unidade
        """,
        "notes": (
            "Não é um documento clínico (sem diagnóstico, sem prescrição), "
            "mas tem elementos de UBS, unidade de saúde. "
            "Expectativa: rejeitar (is_valid=False), mas pode ser aceito pelo score."
        ),
    },
    {
        "filename": "ambig_receita_sem_crm_01.pdf",
        "label": "receita",  # Ground truth: válido (mas incompleto)
        "text": """
Paciente: Renato Braga
Data: 12/05/2026

Amoxicilina 500mg — tomar 1 cápsula de 8 em 8 horas por 7 dias
Dipirona 500mg — tomar 1 comprimido se febre ou dor

Retornar se não melhorar em 3 dias.
        """,
        "notes": (
            "Receita sem CRM visível, sem carimbo, sem CID. "
            "Pode ser receita de UBS onde o médico esqueceu de assinar corretamente. "
            "Em produção: deveria ser rejeitada por incompletude. "
            "Nosso classificador aceita por ter termos clínicos suficientes."
        ),
    },
    {
        "filename": "ambig_laudo_espanhol_01.pdf",
        "label": "laudo",  # Ground truth: válido (documento clínico)
        "text": """
LABORATORIO CENTRAL — RESULTADOS DE ANÁLISIS CLÍNICOS

Paciente: María García   Fecha: 15/05/2026

Hemograma completo:
Eritrocitos: 4,5 × 10⁶/µL   Hemoglobina: 13,8 g/dL
Leucocitos: 6.800 /µL        Plaquetas: 198.000 /µL

Conclusión: Hemograma dentro de parámetros normales.

Dr. Juan Martínez — Médico — Número de registro: 54321
        """,
        "notes": (
            "Laudo em espanhol. Termos similares ao português (eritrocitos/eritrócitos). "
            "OCR pode confundir algumas palavras. "
            "Nosso vocabulário em português pode não capturar bem termos espanhóis."
        ),
    },
    {
        "filename": "ambig_relatorio_nutricional_01.pdf",
        "label": "relatorio_medico",  # Ground truth: válido (clínico, mas limítrofe)
        "text": """
AVALIAÇÃO NUTRICIONAL

Nutricionista: Dra. Camila Ferraz — CRN-PB 12345
Paciente: Sandra Costa   Data: 14/05/2026

Peso: 78 kg   Altura: 1,65m   IMC: 28,7 (Sobrepeso)

Plano alimentar prescrito:
- Café da manhã: 1 fatia de pão integral + 1 ovo mexido
- Almoço: 100g de proteína + salada à vontade + 4 col. arroz integral

Retorno em 30 dias para avaliação de evolução.
        """,
        "notes": (
            "Relatório nutricional tem CRN (nutricionista) não CRM. "
            "É documento clínico para fins da teleconsultoria? "
            "Depende da política do sistema. Nosso classificador aceita."
        ),
    },
    {
        "filename": "ambig_foto_monitor_01.pdf",
        "label": "laudo",  # Ground truth: válido (conteúdo clínico, mas captura de tela)
        "text": """
[IMAGEM: Foto de monitor de computador exibindo resultado de exame]

SISTEMA HOSPITALAR — RESULTADO LABORATORIAL
Paciente: Gustavo Alves

Ureia: 32 mg/dL     Creatinina: 1,1 mg/dL
Glicemia: 88 mg/dL  Sódio: 140 mEq/L
        """,
        "notes": (
            "OCR de foto de monitor: qualidade degradada, pixels de tela visíveis. "
            "O conteúdo é clínico, mas a forma de envio (foto do monitor) "
            "é tecnicamente inaceitável em contexto de teleconsultoria real. "
            "Nosso sistema aceita o conteúdo sem avaliar a forma — limitação importante."
        ),
    },
    {
        "filename": "ambig_texto_minimo_01.pdf",
        "label": "desconhecido",  # Ground truth: inconclusivo
        "text": "Paciente hipertenso. Retorno em 15 dias. Dr. João CRM 11111",
        "notes": (
            "Texto extremamente curto (possível extração parcial de PDF mal formatado). "
            "Tem termos clínicos e CRM, mas sem contexto suficiente. "
            "Nosso sistema rejeita por word_count < 30, aplicando threshold mais alto."
        ),
    },
    {
        "filename": "ambig_formulario_sus_01.pdf",
        "label": "relatorio_medico",  # Ground truth: válido
        "text": """
FICHA DE REFERÊNCIA E CONTRARREFERÊNCIA — SUS

Unidade de origem: UBS João Paulo II
Data: 13/05/2026

Paciente: Eliane Moura   Idade: 58 anos

Motivo do encaminhamento: Investigação de massa pulmonar em RX de tórax
(achado incidental em 10/04/2026).

Exames realizados: Radiografia de tórax PA e perfil (laudo em anexo).

Encaminhar para: Pneumologia / Oncologia torácica

CID provável: R91 — Achado anormal no exame de imagem do pulmão

Médico solicitante: Dr. Fábio Soares — CRM-PB 88888
        """,
        "notes": (
            "Formulário administrativo do SUS com conteúdo clínico. "
            "Deve ser aceito (é parte do fluxo de teleconsultoria). "
            "Nosso classificador aceita corretamente — bom desempenho neste caso."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════
# GERADOR DE PDFS
# ══════════════════════════════════════════════════════════════════════════

def create_pdf(text: str, output_path: Path) -> None:
    """Cria um PDF simples com o texto fornecido."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Converte o texto inteiro de uma vez para latin-1
    safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
    
    pdf.write(6, safe_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
def generate_dataset() -> None:
    """Gera todos os PDFs do dataset de avaliação."""

    manifest = []

    for category, docs, folder in [
        ("valid",     VALID_DOCS,     "valid"),
        ("invalid",   INVALID_DOCS,   "invalid"),
        ("ambiguous", AMBIGUOUS_DOCS, "ambiguous"),
    ]:
        for doc in docs:
            output_path = OUTPUT_DIR / folder / doc["filename"]
            create_pdf(doc["text"], output_path)

            entry = {
                "filename": doc["filename"],
                "category": category,
                "ground_truth_label": doc["label"],
                "ground_truth_is_valid": category == "valid" or (
                    category == "ambiguous" and doc["label"] not in
                    ("desconhecido", "selfie", "fatura", "contrato", "print_tela")
                ),
                "notes": doc.get("notes", ""),
            }
            manifest.append(entry)
            print(f"  ✓ Criado: {folder}/{doc['filename']}")

    # Salvar manifest como JSON
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDataset gerado em: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Total: {len(manifest)} documentos")


if __name__ == "__main__":
    print("Gerando dataset de avaliação...")
    generate_dataset()
