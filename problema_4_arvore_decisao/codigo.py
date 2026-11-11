import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# PASSO 1: CARREGAR DADOS E PRÉ-PROCESSAR (Pergunta 1 e 2)
# ==========================================
print("A carregar os dados de crédito alemão (pode demorar uns segundos)...")
# Usamos o fetch_openml como sugerido no enunciado para obter rótulos legíveis
data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
X = data.data
y = data.target

# Transformar variáveis categóricas (texto) em colunas binárias (One-Hot Encoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# ==========================================
# PASSO 2: DIVISÃO (Pergunta 3)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Nota: Não fazemos StandardScaler porque árvores de decisão não precisam de normalização.

# ==========================================
# PASSO 3: CONTROLO DE COMPLEXIDADE (Pergunta 4)
# ==========================================
profundidades = [2, 3, 5, 7, 10, None]
train_scores, val_scores = [], []

print("A avaliar diferentes profundidades de árvore...")
for prof in profundidades:
    dt = DecisionTreeClassifier(max_depth=prof, random_state=42)
    # cross_validate permite obter as métricas de treino e validação em simultâneo
    scores = cross_validate(dt, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True)
    train_scores.append(scores['train_score'].mean())
    val_scores.append(scores['test_score'].mean())

# Desenhar gráfico de Overfitting
plt.figure(figsize=(8, 5))
prof_labels = [str(p) if p is not None else 'Sem Limite' for p in profundidades]
plt.plot(prof_labels, train_scores, marker='o', label='Acurácia no Treino', color='blue')
plt.plot(prof_labels, val_scores, marker='o', label='Acurácia na Validação (Cross-Val)', color='orange')
plt.title("Acurácia vs. Profundidade da Árvore (max_depth)")
plt.xlabel("Profundidade Máxima")
plt.ylabel("Acurácia Média")
plt.legend()
plt.grid(True)
plt.savefig("grafico_profundidade.png")
print("Gráfico de overfitting guardado como 'grafico_profundidade.png'.")

# Selecionar automaticamente a profundidade que não causa overfitting severo
melhor_idx = val_scores.index(max(val_scores))
melhor_prof = profundidades[melhor_idx]
print(f"-> A melhor profundidade encontrada para generalização foi: {melhor_prof}")

# ==========================================
# PASSO 4: INTERPRETAÇÃO DO MODELO (Pergunta 5)
# ==========================================
print(f"\nA treinar modelo final com max_depth={melhor_prof}...")
modelo_final = DecisionTreeClassifier(max_depth=melhor_prof, random_state=42)
modelo_final.fit(X_train, y_train)

# Obter as 3 variáveis mais importantes
importancias = pd.Series(modelo_final.feature_importances_, index=X_train.columns)
top_3 = importancias.sort_values(ascending=False).head(3)
print("\nOs 3 atributos mais determinantes para o modelo:")
print(top_3)

# Desenhar a própria Árvore
plt.figure(figsize=(16, 8))
plot_tree(modelo_final, filled=True, feature_names=X_train.columns, class_names=modelo_final.classes_, rounded=True, fontsize=10)
plt.title("Árvore de Decisão - Risco de Crédito")
plt.savefig("arvore_decisao.png")
print("\nDiagrama da árvore guardado como 'arvore_decisao.png'.")

# ==========================================
# PASSO 5: AVALIAÇÃO FINA (Pergunta 6)
# ==========================================
y_pred = modelo_final.predict(X_test)
print("\n--- MÉTRICAS FINAIS ---")
print(classification_report(y_test, y_pred))

# Gerar Matriz de Confusão
matriz = confusion_matrix(y_test, y_pred, labels=['good', 'bad'])
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Reds", xticklabels=['Bons Pagadores', 'Maus Pagadores'], yticklabels=['Bons Pagadores', 'Maus Pagadores'])
plt.title("Matriz de Confusão")
plt.xlabel("Previsão do Banco (Modelo)")
plt.ylabel("Situação Real do Cliente")
plt.tight_layout()
plt.savefig("matriz_credito.png")
print("Matriz de confusão guardada como 'matriz_credito.png'.")