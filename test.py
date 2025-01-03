from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Preenchendo valores nulos
data_pd['clean_text'] = data_pd['clean_text'].fillna('')
data_pd['char_count'] = data_pd['char_count'].fillna(0)

# Separando as variáveis independentes (X) e dependente (y)
X = data_pd[['clean_text', 'char_count']]
y = data_pd['ANALISE_RESPOSTA_I']

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurando o pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2)), 'clean_text'),
        ('scaler', StandardScaler(), 'char_count')
    ]
)

# Adicionando uma conversão explícita de colunas 1D para 2D
class ColumnSelector:
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.column_name]]

# Ajustando o pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2)), 'clean_text'),
            ('char_count', StandardScaler(), 'char_count')
        ]
    )),
    ('clf', LogisticRegression(class_weight={0: 3.9, 1: 1}))  # Balanceamento de classes
])

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# Adicionando as probabilidades de classificação à tabela original
data_pd['score'] = pipeline.predict_proba(X)[:, 1]
