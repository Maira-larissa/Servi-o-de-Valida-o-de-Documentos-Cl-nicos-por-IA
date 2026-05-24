# 🏥 Serviço de Validação de Documentos Clínicos por IA

> **Desafio Técnico P04 — Pesquisador(a) em Inteligência Artificial**  
> LAVID/UFPB · Projeto ReNTAI / CLUSTER-19 · Ecossistema V4H de Telessaúde

---

## Índice

1. [Contexto](#contexto)
2. [Abordagem e Justificativa](#abordagem-e-justificativa)
3. [Arquitetura do Serviço](#arquitetura-do-serviço)
4. [Como Executar](#como-executar)
5. [Contrato da API](#contrato-da-api)
6. [Dataset de Avaliação](#dataset-de-avaliação)
7. [Resultados](#resultados)
8. [Análise dos Erros](#análise-dos-erros)
9. [Análise Crítica](#análise-crítica)
10. [Ferramentas de IA Utilizadas](#ferramentas-de-ia-utilizadas)

---

## Contexto

A plataforma **V4H** recebe uploads de documentos enviados por profissionais da Atenção Primária à Saúde (APS) durante solicitações de teleconsultoria. Documentos irrelevantes — selfies, contratos, prints de tela, faturas — sobrecarregam especialistas e prejudicam o fluxo clínico.

Este serviço é o **backend de IA** que classifica automaticamente se um arquivo é um documento clínico legítimo, com:
- Score de confiança (0–1)
- Limiar adaptativo por especialidade
- Justificativa legível da decisão
- Suporte a PDFs digitais e documentos escaneados

---

## Abordagem e Justificativa

### Classificador escolhido: Vocabulário Clínico Ponderado + Heurísticas Estruturais

O texto extraído do documento é comparado contra dois dicionários curados manualmente:

| Componente | Peso | Descrição |
|---|---|---|
| Score clínico | 55% | Soma ponderada de ~110 termos médicos em português (peso 1–3 por especificidade) |
| Score estrutural | 25% | Presença de CRM/CRF, data, CID, padrão de assinatura, nome do paciente |
| Penalidade negativa | 20% | Termos de documentos não-clínicos (nota fiscal, cláusula, screenshot...) |

```
score_final = (0.55 × score_clínico) + (0.25 × score_estrutural) − (0.20 × penalidade)
```

### Por que não foi usado um modelo de linguagem treinável (DistilBERT, TF-IDF)?

Essa decisão foi deliberada:

- **Sem corpus rotulado real disponível.** Treinar com dados sintéticos criaria viés artificial — o modelo aprenderia os próprios exemplos, não a variedade real de documentos da plataforma V4H.
- **F1 inflado, valor real incerto.** Um TF-IDF treinado em 30 documentos sintéticos pode reportar F1 ≥ 0.95 e falhar completamente em documentos reais.
- **Explicabilidade comprometida.** O edital exige `justification` legível. Com vocabulário ponderado, a justificativa é direta: *"Termos 'hemograma', 'leucócitos', 'CRM-PB 12345' encontrados."* Com TF-IDF, são pesos de features abstratos.

**Limitação principal reconhecida:** a abordagem não distingue *texto sobre medicina* (artigo jornalístico) de *documento médico*. Documentado e testado nos casos ambíguos.

### Versão de produção recomendada

Fine-tuning do **BERTimbau** (BERT pré-treinado em 2,68 bilhões de tokens em português) com dataset rotulado de documentos reais do V4H, com threshold calibrado via curva precision-recall levando em conta o custo assimétrico de cada tipo de erro.

---

## Arquitetura do Serviço

```
clinical-doc-validator/
├── app/
│   ├── main.py          # FastAPI — endpoints /validate, /health, /feedback
│   ├── classifier.py    # Classificador com vocabulário ponderado
│   ├── extractor.py     # Extração de texto (PyMuPDF + RapidOCR)
│   ├── threshold.py     # Limiar adaptativo por especialidade e qualidade
│   └── schemas.py       # Contrato de entrada/saída (Pydantic v2)
├── evaluation/
│   ├── generate_test_data.py  # Gera os 30 documentos de teste
│   └── evaluate.py            # Executa avaliação e gera métricas
├── tests/
│   ├── test_classifier.py     # Testes unitários com hipóteses de falha
│   └── test_api.py            # Teste de integração da API
├── .env.example
├── .gitignore
└── requirements.txt
```

### Pipeline de processamento

```
Upload (PDF ou Imagem)
        │
        ▼
┌──────────────────┐
│  Detecção de tipo │  PDF digital / PDF escaneado / Imagem
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 PyMuPDF   RapidOCR
 (digital) (escaneado/imagem)
    │         │
    └────┬────┘
         │ texto normalizado
         ▼
┌──────────────────────┐
│  Classificador       │  score_clínico + score_estrutural − penalidade
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Limiar Adaptativo   │  BASE × fator_especialidade + ajuste_qualidade
└────────┬─────────────┘
         │
         ▼
  { is_valid, score, label, justification, threshold_applied }
```

---

## Como Executar

### Pré-requisitos

- Python 3.11+
- Nenhuma dependência de sistema além do Python (OCR é 100% Python via ONNX)

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/Maira-larissa/Servi-o-de-Valida-o-de-Documentos-Cl-nicos-por-IA.git
cd Servi-o-de-Valida-o-de-Documentos-Cl-nicos-por-IA

# 2. Crie e ative o ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
```

### Executar a API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentação interativa disponível em: **http://localhost:8000/docs**

### Executar os testes

```bash
pytest tests/ -v
```

### Gerar e avaliar o dataset

```bash
# Gerar os 30 documentos de teste
python evaluation/generate_test_data.py

# Executar avaliação (com a API rodando)
python evaluation/evaluate.py
```

---

## Contrato da API

### `POST /validate`

Recebe um arquivo (PDF ou imagem) e retorna a classificação.

**Entrada (multipart/form-data):**

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `file` | File | Sim | PDF ou imagem (JPG, PNG, TIFF, WebP) |
| `specialty` | string | Não | Especialidade clínica (influencia o limiar) |

**Saída (JSON):**

```json
{
  "is_valid": true,
  "confidence_score": 0.6823,
  "label": "laudo",
  "justification": "Documento classificado como laudo/resultado de exame. Termos clínicos identificados: 'hemograma', 'leucócitos', 'diagnóstico', 'crm'. Elementos estruturais presentes (score: 0.67).",
  "threshold_applied": 0.4950,
  "clinical_terms_found": ["hemograma", "leucócitos", "eritrócitos", "diagnóstico", "crm"],
  "extraction_method": "digital_pdf",
  "word_count": 142
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `is_valid` | bool | Decisão final: documento aceito como clínico |
| `confidence_score` | float 0–1 | Score do classificador |
| `label` | string | Tipo detectado: `laudo`, `receita`, `atestado`, `relatorio_medico`, `exame_imagem`, `fatura`, `contrato`, `print_tela`, `selfie`, `desconhecido` |
| `justification` | string | Explicação legível da decisão |
| `threshold_applied` | float 0–1 | Limiar efetivamente usado (adaptativo) |
| `clinical_terms_found` | list | Termos clínicos que influenciaram o score |
| `extraction_method` | string | `digital_pdf` · `ocr_scanned_pdf` · `ocr_image` · `fallback_empty` |

### `GET /health`

Verifica disponibilidade do serviço e retorna o threshold base configurado.

### `POST /feedback`

Registra feedback de falso positivo para ajuste do limiar adaptativo por especialidade.

---

## Limiar Adaptativo

O limiar não é fixo — é calculado dinamicamente:

```
threshold_final = BASE_THRESHOLD × fator_especialidade + ajuste_qualidade
```

**Ajuste por especialidade:**

| Especialidade | Fator | Justificativa |
|---|---|---|
| Oncologia | ×1.10 | Risco clínico máximo; FP pode embasar decisão grave |
| Psiquiatria | ×1.05 | Documentos menos padronizados |
| Cardiologia | ×1.05 | OCR de ECGs pode gerar ruído |
| Neurologia | ×1.05 | Laudos complexos |
| Clínica geral | ×1.00 | Base de comparação |
| Dermatologia | ×0.95 | Documentos fotográficos simples |

**Ajuste por qualidade de extração:**

| Situação | Ajuste | Motivo |
|---|---|---|
| PDF escaneado (OCR) | +0.05 | OCR menos confiável |
| Texto < 30 palavras | +0.10 | Sinal de extração falha |
| Sem texto extraído | → 0.99 | Rejeitar por segurança |

**Histórico de erros:** se a taxa de falsos positivos recente para uma especialidade superar 20%, aplica-se +0.05 adicional (em memória; produção usaria Redis com TTL).

---

## Dataset de Avaliação

### Estrutura (30 documentos sintéticos)

```
evaluation/dataset/
├── valid/       10 documentos clínicos legítimos
├── invalid/     10 documentos não clínicos
└── ambiguous/   10 casos de borda documentados
```

### Categorias

**Válidos (10):**
Laudo de hemograma, receita médica, atestado médico, laudo de tomografia, relatório médico de encaminhamento, laudo de ECG, resultado de glicemia/colesterol, prescrição psiquiátrica, laudo de ultrassom, guia de encaminhamento SUS.

**Inválidos (10):**
Nota fiscal eletrônica, contrato de locação, fatura de energia, boleto bancário, print de WhatsApp, currículo, cardápio de restaurante, proposta comercial, comprovante de residência, manual de produto.

**Ambíguos (10) — os mais importantes:**

| Arquivo | Ground truth | Por que é difícil |
|---|---|---|
| Artigo jornalístico sobre saúde | Inválido | Menciona termos médicos sem ser documento clínico |
| Atestado manuscrito (OCR ruim) | Válido | OCR degrada acentos; vocabulário não bate |
| Exame veterinário | Inválido | Estrutura idêntica ao humano; CRMV ≈ CRM |
| Declaração de comparecimento | Inválido | Menciona UBS mas sem conteúdo clínico |
| Receita sem CRM | Válido | Score estrutural baixo por ausência de registro |
| Laudo em espanhol | Válido | Termos sem acento (eritrocitos) não batem no vocabulário PT |
| Relatório nutricional (CRN) | Válido | CRN ≠ CRM no regex; depende de política |
| Foto de monitor com exame | Válido | Conteúdo clínico mas forma inadequada |
| Texto mínimo extraído | Inválido | Threshold alto por word_count < 30 |
| Formulário SUS de referência | Válido | Documento administrativo com conteúdo clínico |

### Como reproduzir

```bash
python evaluation/generate_test_data.py
# Gera os PDFs em evaluation/dataset/ e o manifest.json com ground truth
```

---

## Resultados

### Calibração do threshold

A primeira rodada de testes revelou um erro de calibração: o threshold inicial de 0.70 foi definido antes de qualquer dado real. Os scores reais mostraram:

- Documentos válidos: **0.48 – 0.68**
- Documentos inválidos: **0.00 – 0.04**

A separação natural está em ~0.10. Qualquer threshold entre 0.05 e 0.48 classifica corretamente os 20 casos claros. O threshold foi recalibrado para **0.45** com base nos dados.

| Threshold testado | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| 0.70 (original) | 0.467 | 0.000 | 0.000 | 0.000 |
| **0.45 (calibrado)** | **0.867** | **0.875** | **0.875** | **0.875** |
| 0.50 | 0.833 | 0.929 | 0.813 | 0.867 |

### Resultados finais (threshold = 0.45)

| Métrica | Valor |
|---|---|
| Acurácia | **0.867** |
| Precisão | **0.875** |
| Recall | **0.875** |
| **F1-Score** | **0.875** |

### Matriz de confusão

|  | Predito: Válido | Predito: Inválido |
|---|---|---|
| **Real: Válido** | TP = 14 | FN = 2 |
| **Real: Inválido** | FP = 2 | TN = 12 |

### Por categoria

| Categoria | Acertos | F1 |
|---|---|---|
| Válidos (10 docs) | 10/10 | 1.000 |
| Inválidos (10 docs) | 10/10 | 1.000 |
| Ambíguos (10 docs) | 6/10 | 0.667 |

---

## Análise dos Erros

### 🔴 Falsos Positivos — documento inválido aceito (2 casos)

**`ambig_artigo_saude_01.pdf`** — Artigo jornalístico sobre hipertensão  
- Score: 0.55 | Label: `desconhecido`  
- O artigo menciona "hipertensão", "Losartana", "cardiologista", "diagnóstico" → score clínico alto.  
- **Causa raiz:** a abordagem não distingue *contexto de uso*. "Losartana" numa receita e "Losartana" numa reportagem têm o mesmo peso no vocabulário.  
- **Risco clínico:** baixo — um especialista identifica imediatamente. Mas revela a limitação central do método.  
- **Solução:** modelo contextual (BERTimbau) distingue "prescrito pelo cardiologista" de "o cardiologista *explica que*".

**`ambig_exame_veterinario_01.pdf`** — Hemograma de cachorro  
- Score: 0.59 | Label: `laudo`  
- Estrutura idêntica ao hemograma humano: eritrócitos, leucócitos, plaquetas, CRM (CRMV capturado pelo regex).  
- **Causa raiz:** vocabulário clínico não distingue medicina humana de veterinária.  
- **Risco clínico:** moderado — em triagem automática, um laudo veterinário pode chegar ao especialista.  
- **Correção simples:** adicionar `"crmv": -5`, `"médica veterinária": -5` ao vocabulário negativo.

### 🟡 Falsos Negativos — documento válido rejeitado (2 casos)

**`ambig_atestado_manuscrito_01.pdf`** — Atestado médico escrito à mão  
- Score: 0.23 | Threshold: 0.80 (penalidade por word_count < 30)  
- OCR retornou texto sem acentos: "medico" em vez de "médico", sem pontuação, CRM ilegível → não bate com o vocabulário acentuado.  
- **Risco clínico:** alto — um atestado urgente pode ser bloqueado por problema de digitalização.  
- **Solução:** adicionar variantes sem acento ao vocabulário (`"medico": 2`, `"diagnostico": 3`) ou normalizar acentos pós-OCR.

**`ambig_relatorio_nutricional_01.pdf`** — Plano alimentar prescrito por nutricionista  
- Score: 0.41 | Threshold: 0.45  
- A nutricionista tem CRN, não CRM. O regex de registro profissional não captura CRN → score estrutural zero. Poucos termos do vocabulário clínico.  
- **Nota de política:** este caso levanta uma questão que vai além do algoritmo — relatórios nutricionais são documentos clínicos válidos para teleconsultoria no V4H? Se sim, o vocabulário e o regex precisam incluir CRN explicitamente.

### Síntese

| Tipo | Quantidade | Causa raiz dominante |
|---|---|---|
| FP (inválido aceito) | 2 | Vocabulário sem contexto semântico |
| FN (válido rejeitado) | 2 | OCR degrada acentuação / regex incompleto |

---

## Análise Crítica

### Quais documentos a abordagem classifica melhor e pior?

**Melhor desempenho:**
- Laudos laboratoriais digitais com terminologia padronizada (hemograma, bioquímica)
- Documentos financeiros — alta densidade de termos negativos exclusivos
- Receitas com CRM, posologia e CID explícitos

**Pior desempenho:**
- Textos *sobre* medicina (artigos, blogs de saúde) — risco de FP
- Documentos de outras áreas da saúde (veterinária, nutrição) — sem distinção
- OCR de documentos manuscritos ou com baixa resolução — FN por perda de acentuação

### Riscos clínicos de falsos positivos vs. falsos negativos

**Falso Positivo** — documento inválido aceito como clínico:
- Especialista recebe lixo na fila de teleconsultoria
- Impacto: sobrecarga, fluxo degradado
- Mitigação: a justificativa e o label mostrados na interface permitem identificação rápida

**Falso Negativo — MAIS GRAVE** — documento clínico válido rejeitado:
- Um laudo oncológico urgente é bloqueado porque o PDF foi mal escaneado
- Impacto: atraso em encaminhamento com potencial consequência clínica séria
- **Decisão de design:** o threshold base (0.45) foi escolhido para favorecer recall. Aceitar um lixo é melhor do que bloquear um exame urgente. Em especialidades de alto risco (oncologia, 0.495), o rigor aumenta, mas a prioridade de recall se mantém.

### O que mudaria em produção?

| Dimensão | Atual | Produção |
|---|---|---|
| **Modelo** | Vocabulário ponderado | Fine-tuned BERTimbau com dados reais do V4H |
| **Dados** | 30 docs sintéticos | ≥ 1.000 docs rotulados + feedback contínuo dos especialistas |
| **OCR** | RapidOCR (ONNX) | Pipeline com pré-processamento de imagem (deskew, denoise, binarização adaptativa) |
| **Threshold** | Calibrado em 30 docs | Curva precision-recall em validation set com custo assimétrico explícito |
| **Infraestrutura** | Síncrono | Fila assíncrona (Celery/RabbitMQ) para PDFs grandes |
| **Monitoramento** | Nenhum | Alertas de drift quando taxa de rejeição se afasta da linha de base |
| **Feedback** | In-memory | Redis com TTL + pipeline de re-treinamento periódico |

---

## Ferramentas de IA Utilizadas

Esta seção é **obrigatória pelo edital** e descreve com precisão onde e como a IA foi usada.

### Ferramenta principal: Claude (Anthropic)

| Parte do trabalho | O que a IA fez | O que foi corrigido ou decidido por mim |
|---|---|---|
| **Estrutura do projeto** | Sugeriu divisão em módulos (`extractor`, `classifier`, `threshold`, `schemas`) | Aceita. Validei contra boas práticas de APIs Python. |
| **Vocabulário clínico** | Gerou lista inicial de termos médicos | Lista misturava inglês com português. Corrigi e adicionei termos do contexto SUS/APS (UBS, COREN, TSH, posologia) ausentes na sugestão. |
| **OCR** | Sugeriu Tesseract inicialmente | Substituí por **RapidOCR** (100% Python, sem binários externos) — decisão minha para compatibilidade cross-platform real. |
| **Regex de normalização** | Sugeriu remover acentos no pré-processamento | **DESCARTADO.** Removeria distinção entre "médico" e "medico", quebrando o vocabulário clínico. Mantive acentos. |
| **Testes unitários** | Gerou esqueleto dos testes | Os casos ambíguos, as "hipóteses de falha explícitas" e os testes de threshold são meus — são o núcleo do ML testing sistemático. |
| **Análise crítica** | Produziu rascunhos genéricos | Reescrevi inteiramente. A análise de riscos clínicos (FP vs FN), os trade-offs do limiar adaptativo e a decisão de priorizar recall em contexto clínico são reflexões minhas. |
| **Calibração do threshold** | Não identificou o problema de calibração | Eu identifiquei ao rodar o `evaluate.py` e ver F1=0 na primeira rodada. A análise dos scores (válidos: 0.48–0.68; inválidos: 0.00–0.04) e a decisão de usar 0.45 foram minhas. |

### Bug introduzido pela IA

A feature `data_formatada` em `classifier.py` recebia o regex de CRM por engano (copy-paste da IA). O bug foi identificado durante revisão do código e corrigido:

```python
# ❌ Regex errado (CRM, não data):
r"\b(crm|crf|coren)[a-z\-\/]*\s*[:\-]?\s*\d{4,8}"

# ✅ Regex correto (data dd/mm/aaaa):
r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"
```

### Avaliação honesta

**Funcionou bem:** scaffolding do FastAPI, Pydantic models, estrutura de arquivos, boilerplate de testes.

**Precisou de correção:** vocabulary em inglês, regex de normalização (acentos), bug de copy-paste no classifier, rascunhos genéricos de análise crítica.

**Descartado:** sugestão de TF-IDF com dados sintéticos (F1 inflado sem valor real), remoção de acentos na normalização, Tesseract como único OCR.

**Fronteira clara:** a decisão técnica central (vocabulário ponderado em vez de ML treinável), a calibração do threshold com dados reais, a análise de riscos clínicos e os casos de borda são decisões e análises minhas. A IA contribuiu com estrutura e código de suporte.

---

## Entrega

Repositório: [github.com/Maira-larissa/Servi-o-de-Valida-o-de-Documentos-Cl-nicos-por-IA](https://github.com/Maira-larissa/Servi-o-de-Valida-o-de-Documentos-Cl-nicos-por-IA)

E-mail: `selecao.rentai@lavid.ufpb.br`  
Assunto: `Desafio ReNTAI 2026 — P04 — Maira Larissa`
