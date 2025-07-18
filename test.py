import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def plot_top_terms_trend(df, date_col, text_col, n_terms=10, days=7):
    """
    Gera o ranking dos top 'n_terms' unigrams/bigrams e plota a volumetria diária dos últimos 'days' dias.
    
    Parâmetros:
    - df: DataFrame com as colunas de data e texto.
    - date_col: Nome da coluna de data.
    - text_col: Nome da coluna de texto.
    - n_terms: Quantos termos mais frequentes considerar.
    - days: Quantos dias anteriores (inclusive) considerar.
    """
    # Garantir tipo datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Definir intervalo de datas
    end_date = df[date_col].max()
    start_date = end_date - pd.Timedelta(days=days-1)
    last_week = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

    # Vetorização (unigramas + bigramas), binary=True para contabilizar por caso
    vectorizer = CountVectorizer(ngram_range=(1,2), binary=True)
    X = vectorizer.fit_transform(last_week[text_col].fillna(''))

    # Frequência total de cada termo
    term_frequencies = np.array(X.sum(axis=0)).flatten()
    terms = np.array(vectorizer.get_feature_names_out())

    # Seleção dos top termos
    top_idx = term_frequencies.argsort()[::-1][:n_terms]
    top_terms = terms[top_idx]

    # Preparar DataFrame de tendência diária
    date_range = pd.date_range(start_date, end_date, freq='D')
    trend_df = pd.DataFrame(index=date_range, columns=top_terms).fillna(0)

    # Preencher contagens diárias
    for term in top_terms:
        col_idx = np.where(terms == term)[0][0]
        counts = X[:, col_idx].toarray().flatten()
        last_week[term] = counts
        daily_counts = last_week.groupby(date_col)[term].sum()
        trend_df.loc[daily_counts.index, term] = daily_counts

    # Plot
    plt.figure(figsize=(10, 6))
    for term in top_terms:
        plt.plot(trend_df.index, trend_df[term], label=term)
    plt.xlabel('Data')
    plt.ylabel('Número de Ocorrências (por caso)')
    plt.title(f'Tendência dos Top {n_terms} Termos nos Últimos {days} Dias')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
# df = pd.read_csv('seus_dados.csv')  # certifique-se de ter colunas 'date' e 'text'
# plot_top_terms_trend(df, date_col='date', text_col='text')
