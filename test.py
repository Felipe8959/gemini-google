import pandas as pd
import re
import unicodedata
from collections import defaultdict

def preprocess_text(text: str) -> list[str]:
    txt = text.lower()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(ch for ch in txt if unicodedata.category(ch) != "Mn")
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    return txt.split()

raw_synonyms = [
    ["gerente","gerencia","responsavel"],
    ["atendimento","suporte","assistencia","apoio"],
    ["ruim","péssimo","descaso","insatisfatorio"],
    ["contato","comunicacao","interacao","conexao","ligacao"],
    ["frustrada","sem exito","sem sucesso","va","vã"],
    ["falta","ausencia","carencia"],
    ["agencia","agência"],
    ["manipulado","manipulacao"],
    ["tentativas","tentativa"]
]

raw_phrases = [
    {"seq": ["falta","de","contato"], "peso": 0.7, "causa": "falta de contato"},
    {"seq": ["tentativas","contato","frustrada"], "peso": 0.8, "causa": "falta de contato"},
]

raw_root_seqs = [
    ["agencia"],
    ["gerente","agencia"],
    ["gerente","da","agencia"]
]

synonyms_to_canonical = {}
for group in raw_synonyms:
    normalized = [preprocess_text(term)[0] for term in group if preprocess_text(term)]
    canonical = normalized[0]
    for tok in normalized:
        synonyms_to_canonical[tok] = canonical

frases_com_pesos = []
for entry in raw_phrases:
    seq_norm = []
    for term in entry["seq"]:
        toks = preprocess_text(term)
        if not toks: continue
        seq_norm.append(synonyms_to_canonical.get(toks[0], toks[0]))
    frases_com_pesos.append({"seq": seq_norm, "peso": entry["peso"], "causa": entry["causa"]})

root_seqs = []
for root in raw_root_seqs:
    seq_norm = []
    for term in root:
        toks = preprocess_text(term)
        if not toks: continue
        seq_norm.append(synonyms_to_canonical.get(toks[0], toks[0]))
    root_seqs.append(seq_norm)

def calcular_score_com_distancia(
    tokens: list[str],
    frases_com_pesos: list[dict],
    root_seqs: list[list[str]],
    synonyms_to_canonical: dict[str, str],
    window_size: int = 10
) -> tuple[float, str | None]:
    posicoes = defaultdict(list)
    for idx, tok in enumerate(tokens):
        canon = synonyms_to_canonical.get(tok, tok)
        posicoes[canon].append(idx)

    best_score = 0.0
    best_causa = None

    for pat in frases_com_pesos:
        seq_canon = [synonyms_to_canonical.get(t, t) for t in pat['seq']]
        pat_positions = [pos for term in seq_canon for pos in posicoes.get(term, [])]
        if not pat_positions:
            continue
        for root in root_seqs:
            root_positions = [pos for term in root for pos in posicoes.get(term, [])]
            if not root_positions:
                continue
            dist = min(abs(i - j) for i in pat_positions for j in root_positions)
            proximity = max(0, window_size - dist) / window_size
            score = pat['peso'] * (1 + proximity)
            if score > best_score:
                best_score, best_causa = score, pat['causa']

    return best_score, best_causa

# Exemplo de aplicação em DataFrame
df = pd.DataFrame({
    "texto": [
        "O gerente da agência não prestou auxílio e faltou contato.",
        "Tivemos várias tentativas de comunicação sem sucesso.",
        "Produto entregue em perfeito estado."
    ]
})

df['tokens'] = df['texto'].apply(preprocess_text)
results = df['tokens'].apply(lambda toks: calcular_score_com_distancia(
    toks, frases_com_pesos, root_seqs, synonyms_to_canonical, window_size=15))
df[['score','causa']] = pd.DataFrame(results.tolist(), index=df.index)

print(df[['texto','score','causa']])
