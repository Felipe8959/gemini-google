import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# 1. Dados
X = data_pd[['clean_text', 'char_coun']]
y = data_pd['ANALISE_RESPOSTA_I']

# 2. Divisão de treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pré-processamento
text_transformer = TfidfVectorizer()
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'clean_text'),
        ('num', num_transformer, 'char_coun')
    ]
)

# 4. Pipeline do modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Treinamento
pipeline.fit(X_train, y_train)

# 6. Avaliação
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
