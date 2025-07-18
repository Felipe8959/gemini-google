import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import zscore

# 1) Carregue seus dados
# Exemplo: df = pd.read_csv('reclamacoes.csv', parse_dates=['date'])
# df deve ter colunas: 'date' (datetime) e 'text' (string)

# Supondo que você já tenha um DataFrame 'df':
df = df.copy()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# 2) Vetorização de n‑grams (bi-gramas e tri-gramas)
vectorizer = CountVectorizer(
    ngram_range=(2,3),    # bi‑grams e tri‑grams
    stop_words=None       # ou liste suas stop‑words em PT
)
X_counts = vectorizer.fit_transform(df['text'])

# 3) Cálculo de TF‑IDF
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# 4) Montar DataFrame de TF‑IDF por documento
tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    index=df.index,
    columns=vectorizer.get_feature_names_out()
)

# 5) Agregação temporal (semanal, diária etc.)
# Aqui usamos semanal ('W'), mas pode ser 'D' para diário
weekly_tfidf = tfidf_df.resample('W').mean()

# 6) Cálculo de z‑score ao longo das semanas
weekly_z = weekly_tfidf.apply(lambda col: zscore(col, nan_policy='omit'))

# 7) Detectar spikes na última semana
threshold = 2.0  # defina seu limiar de z‑score (ex.: 1.5, 2.0, 2.5)
last_week_z = weekly_z.iloc[-1]
spikes = last_week_z[last_week_z > threshold].sort_values(ascending=False)

# 8) Exibir resultados
print("N‑grams em spike (z >", threshold, "):")
print(spikes)
