from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Preenchendo valores nulos
data_pd['clean_text'] = data_pd['clean_text'].fillna('')
data_pd['char_text'] = data_pd['char_text'].fillna(0)

# Criando uma nova coluna que combina clean_text e char_text
data_pd['combined_text'] = data_pd['clean_text'] + " [CHAR_COUNT: " + data_pd['char_text'].astype(str) + "]"

# Configurando o pipeline com a nova coluna
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=100,
        max_features=5000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='saga',
        random_state=42,
        max_iter=1000,
        class_weight={0: 3.9, 1: 1}  # Para desbalanceamento
    ))
])

# Treinando o modelo com a nova coluna
pipeline.fit(data_pd['combined_text'], data_pd['ANALISE_RESPOSTA_I'])

# Fazendo predições
data_pd['score'] = pipeline.predict_proba(data_pd['combined_text'])[:, 1]

# Convertendo de volta para DataFrame do Spark, se necessário
scored_data = spark.createDataFrame(data_pd)
