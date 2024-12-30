from imblearn.pipeline import Pipeline  # Use o pipeline do imblearn
from imblearn.over_sampling import SMOTE

# Atualize o pipeline para incluir o SMOTE
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        max_features=5000,
        sublinear_tf=True
    )),
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),  # Aplica o SMOTE
    ('lr', LogisticRegression(
        penalty='l2',
        C=1.0,  # Regularização
        random_state=6162,
        class_weight='balanced'
    ))
])

# Ajuste o pipeline usando os dados de treino
pipeline.fit(train_data_pd['clean_text'], train_data_pd['ANALISE_RESPOSTA_1'])

# Faça previsões
test_data_pd['predictions'] = pipeline.predict(test_data_pd['clean_text'])
test_data_pd['probabilities'] = pipeline.predict_proba(test_data_pd['clean_text'])[:, 1]
