# Após ajustar o pipeline
pipeline.fit(data_pd['clean_text'], data_pd['ANALISE_RESPOSTA_1'])

# Obtenção das palavras mais importantes
tfidf = pipeline.named_steps['tfidf']
lr = pipeline.named_steps['lr']

# Obtém as palavras e seus pesos
feature_names = tfidf.get_feature_names_out()
coef = lr.coef_[0]

# Ordena por importância
top_features = sorted(zip(coef, feature_names), reverse=True, key=lambda x: abs(x[0]))

# Exibe as 10 palavras mais importantes
print("Palavras mais importantes (positivas):")
for coef, word in top_features[:10]:
    print(f"{word}: {coef}")

print("\nPalavras mais importantes (negativas):")
for coef, word in top_features[-10:]:
    print(f"{word}: {coef}")
