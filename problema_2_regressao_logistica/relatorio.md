
# Relatório - Problema 2: Regressão Logística (Doenças Cardíacas)

## 1. Pré-processamento dos Dados

O conjunto de dados *Heart Disease Cleveland* possui 13 variáveis clínicas e uma variável alvo. Antes do treino, foram realizadas as seguintes etapas:

* **Tratamento de Valores Ausentes:** Os valores marcados como `?` no dataset original foram convertidos para Nulos (NaN) e as respectivas linhas removidas para manter a integridade dos dados clínicos.
* **Binarização do Rótulo:** O diagnóstico original da variável alvo (que variava de 0 a 4) foi binarizado. O valor 0 foi mantido como representação de ausência de doença, e os valores de 1 a 4 foram convertidos para 1, representando a presença de doença cardíaca.
* **Divisão Treino/Teste:** O dataset foi dividido na proporção clássica de 80% dos dados para treino e 20% para teste.
* **Padronização:** As features clínicas, por apresentarem diferentes grandezas numéricas (ex: colesterol em mg/dl vs idade em anos), foram normalizadas com `StandardScaler` (Z-score) para otimizar o processo de convergência do algoritmo de descida de gradiente e regularização.

## 2. Treinamento e Convergência

O modelo foi instanciado utilizando o algoritmo `LogisticRegression` da biblioteca scikit-learn. Atendendo aos requisitos do projeto, utilizou-se o solver `liblinear` (ideal para datasets pequenos e para classificação binária), com a semente aleatória (`random_state`) fixada em 42 e o limite de iterações (`max_iter`) fixado em 1000.

Avaliando o atributo interno do modelo (`n_iter_`), confirmou-se que o algoritmo atingiu a **convergência em 13 iterações**.

## 3. Avaliação de Desempenho

O modelo apresentou uma excelente capacidade de discriminação nos dados de teste:

* **Acurácia (Accuracy):** ~0.90 (Acertou 90% dos diagnósticos gerais).
* **Precisão (Precision):** ~0.90 (Dos que o modelo disse estarem doentes, 90% estavam realmente doentes).
* **Recall (Sensibilidade):** ~0.93 (O modelo conseguiu identificar e encontrar 93% de todos os doentes reais). No contexto médico, um *Recall* alto é essencial, pois o custo de um Falso Negativo (mandar um doente para casa) é crítico.
* **F1-score:** ~0.92 (Média harmônica excelente entre Precisão e Recall).
* **AUC-ROC:** ~0.94 (A área sob a curva indica que o modelo tem 94% de probabilidade de classificar corretamente um paciente doente sorteado aleatoriamente acima de um paciente saudável).
