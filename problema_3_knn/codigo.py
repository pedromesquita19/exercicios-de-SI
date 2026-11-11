import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# PASSO 1: CARREGAR OS DADOS
# ==========================================
print("A carregar os dados de qualidade do vinho...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# O dataset dos vinhos utiliza o ponto e vírgula como separador
df = pd.read_csv(url, sep=';')

# ==========================================
# PASSO 2: PREPARAÇÃO DAS CLASSES (Pergunta 1)
# ==========================================
# Recodificação da variável quality: <=5 (ruim), 6 e 7 (médio), >=8 (bom)
def classificar_qualidade(nota):
    if nota <= 5:
        return 'ruim'
    elif nota <= 7:
        return 'medio'
    else:
        return 'bom'

df['classe'] = df['quality'].apply(classificar_qualidade)

X = df.drop(['quality', 'classe'], axis=1)
y = df['classe']

# ==========================================
# PASSO 3: DIVISÃO E NORMALIZAÇÃO (Perguntas 1 e 2)
# ==========================================
# O stratify=y garante que a proporção de vinhos bons, médios e ruins seja igual no treino e no teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# A Normalização é obrigatória no k-NN!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# PASSO 4: ESCOLHA DO 'k' (Pergunta 3)
# ==========================================
valores_k = [1, 3, 5, 7, 11]
acuracias = []

print("A testar diferentes valores de k usando Validação Cruzada (5 folds)...")
for k in valores_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Testa o modelo 5 vezes para cada k e guarda a média de acerto
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    acuracias.append(scores.mean())

plt.figure(figsize=(7, 4))
plt.plot(valores_k, acuracias, marker='o', linestyle='--', color='green')
plt.title("Acurácia Média vs. Número de Vizinhos (k)")
plt.xlabel("Valor de k")
plt.ylabel("Acurácia Média")
plt.xticks(valores_k)
plt.grid(True)
plt.savefig("grafico_k.png")
print("Gráfico de validação guardado como 'grafico_k.png'.")

# Descobrir automaticamente qual foi o melhor k
melhor_k = valores_k[acuracias.index(max(acuracias))]
print(f"-> O melhor k encontrado foi: {melhor_k}")

# ==========================================
# PASSO 5: AVALIAÇÃO FINAL (Pergunta 4)
# ==========================================
print(f"\nA treinar o modelo final com k={melhor_k}...")
modelo_final = KNeighborsClassifier(n_neighbors=melhor_k)
modelo_final.fit(X_train_scaled, y_train)

y_pred = modelo_final.predict(X_test_scaled)

print("\n--- MÉTRICAS FINAIS ---")
print(classification_report(y_test, y_pred, zero_division=0))

# Gerar e guardar a Matriz de Confusão
matriz = confusion_matrix(y_test, y_pred, labels=['ruim', 'medio', 'bom'])
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Purples", xticklabels=['ruim', 'medio', 'bom'], yticklabels=['ruim', 'medio', 'bom'])
plt.title(f"Matriz de Confusão (k={melhor_k})")
plt.xlabel("Previsão do Modelo")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig("matriz_confusao.png")
print("Matriz de confusão guardada como 'matriz_confusao.png'.")