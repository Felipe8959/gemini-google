# 1. Extrair nomes do TfidfVectorizer
tfidf_names = (
    pipeline.named_steps['preproc']
            .named_transformers_['tfidf']
            .get_feature_names_out()
)

# 2. Defina os nomes manuais das features do extract_text_stats
manual_feature_names = [
    "chars", "num_words", "num_sentences", "avg_word_len", "avg_words_per_sentence", "lexical_diversity",
    "colon_ct", "dash_ct", "qmark_ct", "line_breaks", "list_markers", "uppercase_ratio",
    "stopword_ratio", "spelling_errors", "readability", "has_date", "has_relative_time",
    "has_causa", "has_solucao", "has_prazo", "verb_ratio", "noun_ratio", "pron_ratio",
    "num_contact_times", "num_phone_calls", "contact_success", "contact_fail",
    "email_count", "attachment_flag", "conclusion_ct", "pending_ct", "digit_ratio", "has_cas"
]

# 3. Concatenar todos os nomes de features
feature_names = list(tfidf_names) + manual_feature_names  # Ajuste se tiver FastText ou SBERT também

# 4. Extrair os coeficientes (importância das features)
importance = pipeline.named_steps['clf'].coef_[0]

# 5. Criar DataFrame de importâncias
import pandas as pd

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# 6. Exibir
display(feature_importance_df)
