import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

# Carregar dados
@st.cache
def load_data():
    df = pd.read_excel('base_completa.xlsx', sheet_name='Planilha1')
    return df

df = load_data()

# Sidebar
st.sidebar.title("Configurações de Clusterização")
n_clusters = st.sidebar.slider('Número de Clusters', min_value=2, max_value=10, value=3)

# Pré-processamento
numeric_features = ['Tempo de Jornada (dias)']
categorical_features = [col for col in df.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_preprocessed = preprocessor.fit_transform(df)

# Clusterização
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_preprocessed)
df['Cluster'] = clusters

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed.toarray())

# Visualização dos Clusters com PCA
st.title("Visualização de Clusters")
fig_pca = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df['Cluster'].astype(str),
                      labels={'x': 'PC1', 'y': 'PC2'}, title='Clusters - PCA')
st.plotly_chart(fig_pca)

# Gráfico de Outliers
st.title("Análise de Outliers")
for cluster in df['Cluster'].unique():
    st.subheader(f"Cluster {cluster}")
    cluster_data = df[df['Cluster'] == cluster]
    fig, ax = plt.subplots()
    sns.boxplot(data=cluster_data[numeric_features], ax=ax)
    st.pyplot(fig)

# Correlação
st.title("Correlação entre Variáveis")
fig_corr = plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig_corr)

# Distância entre grupos
st.title("Distância dos Centróides")
st.write("Soma das distâncias ao centroide por cluster:")
st.bar_chart(pd.DataFrame(kmeans.inertia_, columns=["Inertia"]))

# Métrica Silhouette
st.title("Análise de Silhueta")
silhouette_avg = silhouette_score(X_preprocessed, clusters)
st.write(f"Score de Silhueta para {n_clusters} clusters: {silhouette_avg:.2f}")
