import os
import requests
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# PASSO 1: DESCARREGAR E CARREGAR OS DADOS
# ==========================================
url = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
zip_path = "ccpp.zip"
extract_folder = "ccpp_data"

# Faz o download automático se a pasta ainda não existir
if not os.path.exists(extract_folder):
    print("A descarregar o dataset...")
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    os.remove(zip_path)

print("A carregar os dados para a memória...")
# O ficheiro excel está dentro da pasta extraída
file_path = os.path.join(extract_folder, "CCPP", "Folds5x2_pp.xlsx")
df = pd.read_excel(file_path)

# ==========================================
# PASSO 2: ANÁLISE DE CORRELAÇÃO (Perguntas 1 e 2)
# ==========================================
print("A gerar matriz de correlação...")
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação - UTCC")
plt.tight_layout()
plt.savefig("matriz_correlacao.png")
print("Gráfico guardado localmente como 'matriz_correlacao.png'.")

# ==========================================
# PASSO 3: PRÉ-PROCESSAMENTO (Pergunta 3)
# ==========================================
# Separar variáveis independentes (X) e a variável alvo (y)
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Divisão: 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização (Z-score) para deixar todas as variáveis na mesma escala numérica
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# PASSO 4: TREINO E AVALIAÇÃO DO MODELO (Pergunta 4)
# ==========================================
print("A treinar o modelo de Regressão Linear Múltipla...\n")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Fazer previsões com os dados de teste (os 20% separados)
y_pred = model.predict(X_test_scaled)

# Calcular métricas de erro e precisão
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("--- RESULTADOS FINAIS ---")
print(f"R² (Coeficiente de Determinação): {r2:.4f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f} MW")
print(f"MAE (Erro Absoluto Médio): {mae:.4f} MW")