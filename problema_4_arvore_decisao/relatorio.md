
# Relatório - Problema 4: Árvores de Decisão (Risco de Crédito)

## 1. Desbalanceamento e Codificação

O conjunto de dados possui 700 casos de "bons pagadores" (70%) e 300 casos de "maus pagadores" (30%). O impacto disto é que avaliar a acurácia de forma isolada é ilusório: se o modelo prever que *todos* os clientes são bons pagadores sem fazer qualquer análise, ainda terá 70% de acurácia, mas provocará a falência do banco ao ignorar a totalidade das inadimplências reais.
As variáveis categóricas foram tratadas com *One-Hot Encoding*, criando colunas binárias representativas.

## 2. Ausência de Normalização

Ao contrário do k-NN (que calcula distâncias geométricas), as árvores de decisão não necessitam de normalização nem de padronização. O algoritmo divide os dados através de testes isolados (splits) em valores limiares (ex: *Idade > 25* ou *Crédito < 1000*). A escala numérica de um atributo não afeta em nada a posição matemática do corte.

## 3. Controle de Complexidade (Underfitting e Overfitting)

Através da validação cruzada, analisou-se o impacto da profundidade da árvore (`max_depth`):

* **Profundidades Baixas (ex: 2):** Ocorreu *underfitting*. O modelo desenhou regras demasiado generalistas e teve mau desempenho tanto no treino como no teste.
* **Profundidades Altas (ex: 10 ou Sem Limite):** Ocorreu *overfitting* claro. A acurácia de treino aproximou-se dos 100% (o modelo decorou cada cliente), mas o desempenho na validação desabou porque a árvore se tornou num emaranhado caótico sem capacidade de generalização.
* **Gini vs Entropia:** Experimentalmente, a troca do critério de Gini para Entropia gera alterações pontuais nas fronteiras, mas pouca ou nenhuma diferença estatística relevante na métrica agregada deste dataset.

## 4. Interpretabilidade

Os atributos mais determinantes para o modelo, de acordo com o índice de Gini, giram sempre em torno do estado financeiro imediato do cliente (o estado da conta corrente `checking_status` é tradicionalmente a raiz da árvore) e do valor do empréstimo.
A árvore desenhada comprova a sua transparência perante a LGPD: um auditor financeiro consegue seguir graficamente o percurso de decisão a partir da raiz até uma folha, criando uma regra legível em português do tipo: *"Se o cliente não possui conta corrente e solicita um empréstimo superior a X, então é classificado como mau pagador"*.

## 5. Erro Frequente e Ajustes de Treino

Na aplicação comercial de concessão de crédito, **o erro mais grave é o Falso Positivo (prever um "mau pagador" como "bom pagador")**. Trata-se de dinheiro emprestado que não vai regressar ao banco, gerando prejuízo financeiro direto e imediato.
Para reduzir este erro no treino, o melhor ajuste seria balancear os pesos das classes utilizando o parâmetro `class_weight='balanced'` na instância da `DecisionTreeClassifier`. Isto obriga o algoritmo a penalizar financeiramente/matematicamente os erros cometidos na classe minoritária (os 30% de maus pagadores).
