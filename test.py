# ========================================================
# EXEMPLO DE SCRIPT PARA PREVER 'PRODUTO_FEBRABAN' E 'MOTIVO_FEBRABAN'
# ========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1) Ler a planilha / CSV
# --------------------------------------------------------
# Ajuste o "sep" se necessário, dependendo do delimitador do seu arquivo
df = pd.read_csv('base.csv', sep=';', encoding='utf-8')

# 2) Criar a coluna de texto unificada (exemplo)
# --------------------------------------------------------
# Se houver valores ausentes em alguma dessas colunas, usamos fillna('') para evitar problemas
df['texto_completo'] = (
    df['DESCR FAMILIA'].fillna('') + ' ' +
    df['DESCR PRODUTO'].fillna('') + ' ' +
    df['DESCR ASSUNTO'].fillna('')
)

# 3) Separar em treino e teste (para PRODUTO_FEBRABAN)
# --------------------------------------------------------
X_produto = df['texto_completo']
y_produto = df['PRODUTO_FEBRABAN']  # Alvo 1

# Divisão treino/teste
X_train_produto, X_test_produto, y_train_produto, y_test_produto = train_test_split(
    X_produto, y_produto, 
    test_size=0.3,         # 30% para teste, 70% para treino
    random_state=42,       # semente fixa para reproduzibilidade
    stratify=y_produto     # mantém proporção de classes
)

# 4) Vetorização com TF-IDF (para PRODUTO_FEBRABAN)
# --------------------------------------------------------
vectorizer_produto = TfidfVectorizer(
    max_features=5000,     # limite de 5.000 termos (ajuste conforme necessário)
    ngram_range=(1,2),     # unigrams e bigrams (opcional)
    stop_words=None        # pode definir uma lista de stopwords para PT-BR, se quiser
)

X_train_produto_tfidf = vectorizer_produto.fit_transform(X_train_produto)
X_test_produto_tfidf = vectorizer_produto.transform(X_test_produto)

# 5) Modelo para PRODUTO_FEBRABAN (Logistic Regression)
# --------------------------------------------------------
model_produto = LogisticRegression(max_iter=1000)
model_produto.fit(X_train_produto_tfidf, y_train_produto)

# 6) Avaliação do modelo (PRODUTO_FEBRABAN)
# --------------------------------------------------------
y_pred_produto = model_produto.predict(X_test_produto_tfidf)
print("=== CLASSIFICAÇÃO PARA 'PRODUTO_FEBRABAN' ===")
print(classification_report(y_test_produto, y_pred_produto))

# --------------------------------------------------------
# *REPITA* o processo para 'MOTIVO_FEBRABAN'
# --------------------------------------------------------

# 3b) Separar em treino e teste (para MOTIVO_FEBRABAN)
X_motivo = df['texto_completo']
y_motivo = df['MOTIVO_FEBRABAN']  # Alvo 2

X_train_motivo, X_test_motivo, y_train_motivo, y_test_motivo = train_test_split(
    X_motivo, y_motivo,
    test_size=0.3,
    random_state=42,
    stratify=y_motivo
)

# 4b) Vetorização com TF-IDF (para MOTIVO_FEBRABAN)
# (Podemos criar um TF-IDF separado, pois o conjunto de classes e distribuição de dados
#  podem ser diferentes. Em alguns casos, você pode reaproveitar o mesmo, mas aqui vamos
#  demonstrar como seria rodar separado.)
vectorizer_motivo = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words=None
)

X_train_motivo_tfidf = vectorizer_motivo.fit_transform(X_train_motivo)
X_test_motivo_tfidf = vectorizer_motivo.transform(X_test_motivo)

# 5b) Modelo para MOTIVO_FEBRABAN (Logistic Regression)
model_motivo = LogisticRegression(max_iter=1000)
model_motivo.fit(X_train_motivo_tfidf, y_train_motivo)

# 6b) Avaliação do modelo (MOTIVO_FEBRABAN)
y_pred_motivo = model_motivo.predict(X_test_motivo_tfidf)
print("=== CLASSIFICAÇÃO PARA 'MOTIVO_FEBRABAN' ===")
print(classification_report(y_test_motivo, y_pred_motivo))

# ========================================================
# FIM DO SCRIPT
# ========================================================
