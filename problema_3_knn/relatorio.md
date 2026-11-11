
# Relatório - Problema 3: Classificação de Vinhos (k-NN)

## 1. Preparação e Justificação da Normalização

A variável `quality` foi recodificada com sucesso em três categorias: ruim ($\le5$), médio (6-7) e bom ($\ge8$).
Para o k-NN, **a normalização dos atributos é uma etapa matemática obrigatória**. Como o algoritmo classifica uma amostra medindo a distância Euclidiana geométrica entre os pontos, variáveis com uma escala muito grande (como o dióxido de enxofre, que pode chegar aos 289 $mg/L$) iriam dominar o cálculo face a variáveis de escala pequena (como os cloretos, na casa dos 0.08 $g/dm^3$). A normalização por *Z-score* colocou tudo na mesma escala geométrica.

## 2. Divisão Treino/Teste

Os dados foram divididos na proporção de 80% para treino e 20% para teste. Foi utilizado o parâmetro de estratificação (`stratify`) para assegurar que a proporção original muito desequilibrada das classes de vinhos (onde os vinhos "médios" são a esmagadora maioria) fosse espelhada com exatidão tanto no conjunto de treino como no conjunto de teste.

## 3. Treino, Escolha do k e Impacto dos Valores Extremos

Através de uma validação cruzada de 5 folds sobre os valores de k (1, 3, 5, 7 e 11), escolheu-se o melhor `k` com base na maior acurácia média atingida.

* **Valores de k muito baixos (ex: k=1):** O modelo sofre de *overfitting* (sobreajuste). Ele fica demasiado sensível a ruídos e aos "outliers" químicos isolados, criando fronteiras de decisão instáveis.
* **Valores de k muito altos:** O modelo sofre de *underfitting* (subajuste). Ele passa a englobar tantos vizinhos que perde a capacidade de identificar os padrões locais de qualidade, limitando-se muitas vezes a classificar qualquer amostra como a classe maioritária ("médio").

## 4. Avaliação e Análise de Erros Frequentes

Através da Matriz de Confusão gerada sobre os dados de teste, nota-se que o modelo atinge o seu melhor desempenho nas classes "médio" e "ruim". No entanto, **o modelo erra com mais frequência entre as classes "ruim" e "médio"**, classificando muitas vezes um vinho ruim como sendo médio.
Isto acontece porque a transição química entre uma nota 5 (ruim) e uma nota 6 (médio) é extremamente sutil na vida real, e como a classe "médio" contém a vasta maioria dos exemplos de treino, o algoritmo fica estatisticamente enviesado para a escolher em casos de dúvida. Além disso, a classe "bom" tem pouquíssimas amostras, dificultando a aprendizagem do padrão da alta qualidade.
