import os
import glob
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

K = 3  # número de grupos escolhido (ver gráfico da tarefa 3)

# ==========================================
# DOWNLOAD AUTOMÁTICO (só baixa se ainda não existir)
# ==========================================
url = "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip"

if not os.path.exists("wholesale_data"):
    resposta = requests.get(url)
    with open("wholesale.zip", "wb") as arquivo:
        arquivo.write(resposta.content)
    with zipfile.ZipFile("wholesale.zip") as zip_ref:
        zip_ref.extractall("wholesale_data")
    os.remove("wholesale.zip")

# procura o arquivo .csv dentro da pasta extraída
caminho_csv = glob.glob("wholesale_data/**/*.csv", recursive=True)[0]
df = pd.read_csv(caminho_csv)

# ==========================================
# TAREFA 1: PREPARAÇÃO DOS DADOS
# ==========================================
# Channel e Region ficam de lado (só serão usados na validação externa)
canal = df["Channel"]
regiao = df["Region"]
gastos = df.drop(columns=["Channel", "Region"])   # só as 6 categorias de gasto

# Estatísticas descritivas
estatisticas = gastos.describe().T
print(estatisticas[["mean", "50%", "std", "min", "max"]].round(1))

# Passo (a): logaritmo, x' = log(1 + x), reduz assimetria e outliers
gastos_log = np.log1p(gastos)

# Passo (b): Z-score, deixa todas as categorias com média 0 e desvio 1
scaler = StandardScaler()
X_pad = scaler.fit_transform(gastos_log)   # dados prontos para o k-Means

# Boxplots antes e depois do log, lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
gastos.boxplot(ax=ax1)
ax1.set_title("Original")
ax1.tick_params(axis="x", labelrotation=30)
gastos_log.boxplot(ax=ax2)
ax2.set_title("Depois do log")
ax2.tick_params(axis="x", labelrotation=30)
plt.tight_layout()
plt.savefig("boxplot.png")
plt.close()

# Correlação entre as categorias
correlacao = gastos.corr()
print("\nCorrelação:")
print(correlacao.round(2))
print("\nGrocery x Detergents_Paper:", round(correlacao.loc["Grocery", "Detergents_Paper"], 2))

# ==========================================
# TAREFA 3: ESCOLHA DO K
# ==========================================
valores_k = range(2, 11)
inercias = []
silhuetas = []

for k in valores_k:
    modelo_k = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    modelo_k.fit(X_pad)

    inercia = modelo_k.inertia_
    silhueta = silhouette_score(X_pad, modelo_k.labels_)

    inercias.append(inercia)
    silhuetas.append(silhueta)
    print(f"k={k}  inércia={inercia:.1f}  silhueta={silhueta:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(valores_k, inercias, marker="o")
ax1.set_title("Cotovelo (inércia)")
ax1.set_xlabel("k")
ax2.plot(valores_k, silhuetas, marker="o")
ax2.set_title("Silhueta")
ax2.set_xlabel("k")
plt.tight_layout()
plt.savefig("escolha_k.png")
plt.close()

# ==========================================
# TAREFA 4: INICIALIZAÇÃO (10 sementes diferentes)
# ==========================================
inercias_random = []
inercias_plus = []

for semente in range(10):
    modelo_random = KMeans(n_clusters=K, init="random", n_init=1, random_state=semente)
    modelo_random.fit(X_pad)
    inercias_random.append(modelo_random.inertia_)

    modelo_plus = KMeans(n_clusters=K, init="k-means++", n_init=1, random_state=semente)
    modelo_plus.fit(X_pad)
    inercias_plus.append(modelo_plus.inertia_)

print("\nrandom    :", np.round(inercias_random, 1))
print("          desvio =", round(np.std(inercias_random), 3))
print("k-means++ :", np.round(inercias_plus, 1))
print("          desvio =", round(np.std(inercias_plus), 3))

# ==========================================
# TAREFA 5: MODELO FINAL E INTERPRETAÇÃO
# ==========================================
modelo = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
modelo.fit(X_pad)
grupos = modelo.labels_   # grupo (0, 1 ou 2) de cada cliente

# Centróides voltam para a escala original em 3 passos
centroides_padronizados = modelo.cluster_centers_                    # estão em Z-score
centroides_log = scaler.inverse_transform(centroides_padronizados)   # desfaz o Z-score
centroides_originais = np.expm1(centroides_log)                      # desfaz o log

centroides = pd.DataFrame(centroides_originais, columns=gastos.columns)
print("\nCentróides (escala original):")
print(centroides.round(0))

# Gráfico de barras: .T troca linhas e colunas (categorias no eixo x, um grupo por cor)
centroides.T.plot(kind="bar", figsize=(10, 5), title="Centróides por categoria")
plt.tight_layout()
plt.savefig("centroides.png")
plt.close()

# Validação externa: compara os grupos com Channel e Region
# Channel: 1 = Horeca, 2 = Retail | Region: 1 = Lisboa, 2 = Porto, 3 = Outras
print("\nGrupo x Channel:")
print(pd.crosstab(grupos, canal))
print("\nGrupo x Region:")
print(pd.crosstab(grupos, regiao))

ari_canal = adjusted_rand_score(canal, grupos)
ari_regiao = adjusted_rand_score(regiao, grupos)
print("\nARI com Channel:", round(ari_canal, 3))
print("ARI com Region :", round(ari_regiao, 3))

# Quais categorias mais separam Horeca de Retail?
medias_por_canal = gastos_log.groupby(canal).mean()          # média do log em cada canal
diferenca = medias_por_canal.loc[2] - medias_por_canal.loc[1]   # Retail - Horeca
diferenca = diferenca.sort_values(ascending=False)
print("\nRetail - Horeca:")
print(diferenca.round(2))

# PCA: reduz as 6 dimensões a 2 só para poder desenhar
pca = PCA(n_components=2)
pontos_2d = pca.fit_transform(X_pad)
centroides_2d = pca.transform(modelo.cluster_centers_)
print("\nVariância explicada pelos 2 componentes:", pca.explained_variance_ratio_.round(3))

plt.scatter(pontos_2d[:, 0], pontos_2d[:, 1], c=grupos, cmap="viridis", alpha=0.8)
plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1], c="red", marker="X", s=200)
plt.title("Clientes em 2D (PCA)")
plt.savefig("pca.png")
plt.close()