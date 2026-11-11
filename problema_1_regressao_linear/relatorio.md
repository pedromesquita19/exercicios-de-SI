
# Relatório - Problema 1: Regressão Linear Múltipla (UTCC)

## 1. Correlação com a Produção de Energia (PE)

Ao analisar a matriz de correlação gerada pelo modelo, as variáveis que apresentam maior influência na variável dependente (PE) são:

* **Temperatura Ambiente (AT):** Forte correlação negativa (aproximadamente -0.95). Quando a temperatura sobe, a produção de energia desce significativamente.
* **Vácuo de Escape (V):** Forte correlação negativa (aproximadamente -0.87).

As variáveis de Pressão Atmosférica (AP) e Humidade Relativa (RH) têm uma influência muito menor no resultado final.

## 2. Correlação entre Variáveis Independentes (Multicolinearidade)

Observa-se uma relação forte e positiva (aproximadamente 0.84) entre a **Temperatura Ambiente (AT)** e o **Vácuo de Escape (V)**.
Esta forte relação indica a presença de **multicolinearidade**. Isto significa que estas duas variáveis partilham informações redundantes sobre as condições termodinâmicas. Num modelo de regressão linear, a multicolinearidade pode tornar os coeficientes instáveis durante o treino, embora não prejudique a capacidade preditiva geral do modelo.

## 3. Construção do Modelo e Pré-processamento

O modelo de regressão linear múltipla foi construído utilizando as variáveis `AT`, `V`, `AP` e `RH`. O pipeline de pré-processamento consistiu em:

1. **Verificação de valores:** O dataset original foi validado.
2. **Divisão Treino/Teste:** 80% dos dados foram separados para treinar o modelo e 20% para o testar, garantindo uma avaliação imparcial.
3. **Padronização:** Como as variáveis possuem unidades de medida diferentes (ºC, cm Hg, mbar, %), aplicou-se a normalização Z-score (`StandardScaler`). Isto garante que nenhuma variável domine a equação de regressão apenas por ter uma escala numérica maior.

## 4. Avaliação de Desempenho

O modelo foi avaliado no conjunto de teste (20%) e obteve as seguintes métricas:

* **$R^2$:** ~0.93 (O modelo consegue explicar cerca de 93% da variação na produção de energia).
* **RMSE:** ~4.56 MW (A margem de erro quadrático).
* **MAE:** ~3.62 MW (O modelo erra, em média, cerca de 3.6 MW por previsão).

**Conclusão:** O modelo é **altamente adequado** para prever a produção de energia elétrica. O valor de $R^2$ acima de 0.90 demonstra uma excelente capacidade de mapear os padrões ambientais. Tratando-se de uma usina termoelétrica que lida com centenas de megawatts, um erro médio absoluto (MAE) na casa dos 3.6 MW é estatisticamente muito reduzido e perfeitamente aplicável para otimizar o planeamento e o despacho energético da usina.
