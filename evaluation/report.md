# Relatório de Avaliação — Classificador de Documentos Clínicos

## Tabela de Resultados Globais

| Métrica   | Valor  |
|-----------|--------|
| Acurácia  | 0.8333 |
| Precisão  | 0.8667 |
| Recall    | 0.8125 |
| F1-Score  | 0.8387 |

### Matriz de Confusão

| | Predito: Válido | Predito: Inválido |
|---|---|---|
| **Real: Válido**   | TP=13 | FN=3 |
| **Real: Inválido** | FP=2 | TN=12 |

## Erros Mais Representativos

### `ambig_artigo_saude_01.pdf`
**Tipo de erro:** 🔴 Falso Positivo (doc inválido aceito)  
**Confiança:** 0.55 | **Limiar:** 0.45  
**Análise:** Caso ambíguo: menciona termos clínicos (hipertensão, infarto, Losartana, cardiologista, diagnóstico) mas é texto jornalístico. Nosso classificador tende a aceitar erroneamente. Limitação conhecida: sem modelo de contexto, não distingue 'texto sobre medicina' de 'documento médico'.

### `ambig_atestado_manuscrito_01.pdf`
**Tipo de erro:** 🟡 Falso Negativo (doc válido rejeitado)  
**Confiança:** 0.23 | **Limiar:** 0.55  
**Análise:** OCR de atestado manuscrito degrada a qualidade do texto. Sem acentos, sem pontuação, CRM parcialmente ilegível. Nosso classificador pode rejeitar por score baixo.

### `ambig_exame_veterinario_01.pdf`
**Tipo de erro:** 🔴 Falso Positivo (doc inválido aceito)  
**Confiança:** 0.59 | **Limiar:** 0.45  
**Análise:** Exame veterinário tem estrutura idêntica ao humano: hemograma, leucócitos, eritrócitos, assinatura com registro profissional. Diferença: CRMV (veterinário) vs CRM (humano). Nosso classificador não distingue — limitação séria em contexto de saúde humana.

### `ambig_receita_sem_crm_01.pdf`
**Tipo de erro:** 🟡 Falso Negativo (doc válido rejeitado)  
**Confiança:** 0.45 | **Limiar:** 0.45  
**Análise:** Receita sem CRM visível, sem carimbo, sem CID. Pode ser receita de UBS onde o médico esqueceu de assinar corretamente. Em produção: deveria ser rejeitada por incompletude. Nosso classificador aceita por ter termos clínicos suficientes.

### `ambig_relatorio_nutricional_01.pdf`
**Tipo de erro:** 🟡 Falso Negativo (doc válido rejeitado)  
**Confiança:** 0.41 | **Limiar:** 0.45  
**Análise:** Relatório nutricional tem CRN (nutricionista) não CRM. É documento clínico para fins da teleconsultoria? Depende da política do sistema. Nosso classificador aceita.
