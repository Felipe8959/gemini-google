import matplotlib.pyplot as plt

# ----------------------------------------
# Assumindo que você já tenha calculado:
# - weekly_tfidf: DataFrame (índice datetime) com TF‑IDF médio de cada n‑gram por período
# - weekly_z: DataFrame (mesmo índice) com z‑scores de cada n‑gram por período
# - spikes: Series com z‑score da última semana, indexada por n‑gram
# ----------------------------------------

# Número de n‑grams que você quer monitorar
top_n = 5

# Seleciona os top N n‑grams em spike na última semana
top_spikes = spikes.nlargest(top_n).index.tolist()

# 1) Gráficos de linha — evolução do TF‑IDF médio
for ngram in top_spikes:
    plt.figure()
    plt.plot(
        weekly_tfidf.index,
        weekly_tfidf[ngram],
        marker='o',
        linestyle='-'
    )
    plt.title(f'Evolução TF‑IDF médio: "{ngram}"')
    plt.xlabel('Período')
    plt.ylabel('TF‑IDF médio')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

# 2) Gráfico de barras — z‑score dos top N na última semana
plt.figure()
values = spikes[top_spikes]
plt.bar(top_spikes, values)
plt.title('Z‑score dos Top N‑grams na Última Semana')
plt.xlabel('N‑gram')
plt.ylabel('Z‑score')
plt.xticks(rotation=45)
plt.axhline(y=threshold, linestyle='--', linewidth=1)
plt.text(
    x=0, y=threshold + 0.1,
    s=f'Threshold = {threshold}',
    va='bottom'
)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Exibe todos os gráficos
plt.show()
