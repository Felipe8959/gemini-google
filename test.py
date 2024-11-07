import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Carregar dados
@st.cache
def load_data():
    df = pd.read_excel('base_completa.xlsx', sheet_name='Planilha1')
    return df

df = load_data()

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
n_clusters = 3  # Número de clusters fixo ou ajustável via sidebar
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_preprocessed)
df['Cluster'] = clusters

# Características por cluster
st.title("Características dos Clusters")

for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    
    # Estatísticas do cluster
    cluster_summary = {
        "Cluster": cluster,
        "Tamanho": len(cluster_data),
        "Tempo Médio de Jornada": cluster_data['Tempo de Jornada (dias)'].mean(),
        "Satisfação Mais Comum": cluster_data['Satisfação'].mode()[0],
        "Motivo Mais Comum": cluster_data['Motivo'].mode()[0]
    }
    
    # Exibir Card
    with st.container():
        st.markdown(f"### Cluster {cluster}")
        st.markdown(f"**Tamanho**: {cluster_summary['Tamanho']}")
        st.markdown(f"**Tempo Médio de Jornada**: {cluster_summary['Tempo Médio de Jornada']:.2f} dias")
        st.markdown(f"**Satisfação Mais Comum**: {cluster_summary['Satisfação']}")
        st.markdown(f"**Motivo Mais Comum**: {cluster_summary['Motivo Mais Comum']}")
        st.markdown("---")
