import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Carregando dados de exemplo
df = pd.read_csv('base.csv', sep=';', encoding='utf-8')

# 2. Unindo colunas de texto em uma única (exemplo)
df['texto_completo'] = (
    df['DESCR FAMILIA'].fillna('') + ' ' +
    df['DESCR PRODUTO'].fillna('') + ' ' +
    df['DESCR ASSUNTO'].fillna('')
)

# 3. Separando features (X) e alvo (y) para PRODUTO_FEBRABAN
X_produto = df['texto_completo']
y_produto = df['PRODUTO_FEBRABAN']

X_train, X_test, y_train, y_test = train_test_split(
    X_produto, y_produto,
    test_size=0.3,
    random_state=42,
    stratify=y_produto
)

# 4. Criando uma Pipeline que faz TF-IDF e depois usa LogisticRegression
pipeline_produto = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])

# 5. Treinando a pipeline
pipeline_produto.fit(X_train, y_train)

# 6. Avaliando rapidamente (PRODUTO_FEBRABAN)
y_pred = pipeline_produto.predict(X_test)
print("=== CLASSIFICAÇÃO PARA 'PRODUTO_FEBRABAN' ===")
print(classification_report(y_test, y_pred))

# 7. Salvando a pipeline inteira em um arquivo .pkl
joblib.dump(pipeline_produto, 'pipeline_produto.pkl')
print("Arquivo 'pipeline_produto.pkl' salvo com sucesso!")
