import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# ==========================================
# PASSO 1: CARREGAR OS DADOS
# ==========================================
print("A baixar os dados clínicos...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# O dataset original usa '?' para indicar valores ausentes
df = pd.read_csv(url, names=columns, na_values='?')

# ==========================================
# PASSO 2: PRÉ-PROCESSAMENTO
# ==========================================
# 1. Tratamento de valores ausentes (removendo as poucas linhas com NaN)
df = df.dropna()

# 2. Binarização da variável alvo: 
# O dataset original tem valores de 0 a 4. Transformamos 0 em 'sem doença' e >0 em 'com doença' (1).
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 3. Separar variáveis independentes (X) e alvo (y)
X = df.drop('target', axis=1)
y = df['target']

# 4. Divisão Treino/Teste (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Padronização (Z-score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# PASSO 3: TREINO DO MODELO
# ==========================================
print("A treinar Regressão Logística...")
# Parâmetros exigidos no problema 2
model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Capturar o número de iterações necessárias para convergir
iteracoes = model.n_iter_[0]
print(f"O modelo convergiu em {iteracoes} iterações.\n")

# ==========================================
# PASSO 4: AVALIAÇÃO DO MODELO
# ==========================================
y_pred = model.predict(X_test_scaled)
# Obter probabilidades para calcular a curva ROC (coluna 1 = probabilidade de ter doença)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("--- MÉTRICAS DE AVALIAÇÃO ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")

# Gerar e salvar a Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Curva ROC - Previsão de Doenças Cardíacas')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("curva_roc.png")
print("\nCurva ROC salva localmente como 'curva_roc.png'.")