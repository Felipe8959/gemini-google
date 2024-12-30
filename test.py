from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
        # Caso deseje testar remoção de stopwords:
        # stop_words='portuguese'
    )),
    ("clf", LogisticRegression(
        penalty='l2',
        C=1.0,               # regularização mediana
        solver='lbfgs',
        random_state=42,
        class_weight='balanced'  # útil se houver desbalanceamento
    ))
])
