from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Preenchendo valores nulos
data_pd['clean_text'] = data_pd['clean_text'].fillna('')
data_pd['char_text'] = data_pd['char_text'].fillna(0)

# Garantir que as colunas sejam arrays 2D
X = data_pd[['clean_text', 'char_text']]  # Seleciona as colunas como DataFrame
y = data_pd['ANALISE_RESPOSTA_I']         # Coluna target

# Criando um transformador de colunas
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=100,
            max_features=5000,
            sublinear_tf=True
        ), 'clean_text'),
        ("numeric", StandardScaler(), 'char_text')  # Normalização da variável numérica
    ]
)

# Configurando o pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='saga',
        random_state=42,
        max_iter=1000,
        class_weight={0: 3.9, 1: 1}  # Para desbalanceamento
    ))
])

# Treinando o modelo
pipeline.fit(X, y)

# Fazendo predições
data_pd['score'] = pipeline.predict_proba(X)[:, 1]

# Convertendo de volta para DataFrame do Spark, se necessário
scored_data = spark.createDataFrame(data_pd)
