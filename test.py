

# =====================================
# 4) DEFININDO O NÚMERO DE CLUSTERS (K)
# =====================================

# Método do "cotovelo" (Elbow method) para ter uma ideia de qual K pode ser interessante
wcss = []
K_values = range(2, 8)  # Vamos testar de 2 até 7 clusters

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_cluster)
    wcss.append(kmeans.inertia_)

# Plot do Elbow
plt.figure(figsize=(6,4))
plt.plot(K_values, wcss, marker='o')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()


# =====================================
# 5) AVALIAÇÃO COM MÉTRICA DE SILHUETA
# =====================================

silhouette_scores = {}
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_cluster)
    silhouette_scores[k] = silhouette_score(df_cluster, labels)

for k, v in silhouette_scores.items():
    print(f"K = {k} -> Silhouette Score: {v:.4f}")

# Exemplo de visualização do Silhouette Score
plt.figure(figsize=(6,4))
plt.bar(silhouette_scores.keys(), silhouette_scores.values())
plt.title('Silhouette Score por número de Clusters')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# =====================================
# 6) ESCOLHA DO NÚMERO DE CLUSTERS E TREINAMENTO FINAL
# =====================================

# Suponha que, analisando os gráficos e métricas, decidimos por K=3
# (isso é um exemplo; escolha com base em suas análises)

k_final = 3
kmeans_final = KMeans(n_clusters=k_final, random_state=42)
df['cluster'] = kmeans_final.fit_predict(df_cluster)

# Verificando a quantidade de registros em cada cluster
df['cluster'].value_counts()

# =====================================
# 7) ANÁLISE DOS CLUSTERS ENCONTRADOS
# =====================================

# Podemos fazer estatísticas descritivas por cluster
descricao_por_cluster = df.groupby('cluster').agg({
    'idade': ['mean', 'median'],
    'renda': ['mean', 'median'],
    'divida': ['mean', 'median'],
    'uf': lambda x: x.value_counts().index[0],     # Moda da UF
    'sexo': lambda x: x.value_counts().index[0],   # Moda do Sexo
    'segmento': lambda x: x.value_counts().index[0]# Moda do Segmento
})

descricao_por_cluster

# =====================================
# 8) VISUALIZANDO OS CLUSTERS (EXEMPLO)
# =====================================

# Nota: Visualizar clusters multidimensionais em 2D/3D pode exigir redução de dimensionalidade (ex.: PCA).
# Vamos apenas demonstrar um PCA para plotar em 2D:

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_cluster)
df['pca1'] = pca_result[:,0]
df['pca2'] = pca_result[:,1]

# Plot
plt.figure(figsize=(6,4))
for c in range(k_final):
    plt.scatter(df[df['cluster'] == c]['pca1'],
                df[df['cluster'] == c]['pca2'],
                label=f'Cluster {c}')
plt.title('Visualização dos Clusters em 2D (via PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
