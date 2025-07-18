import re
import unicodedata
from collections import defaultdict
import pandas as pd

# 1) Função de pré‑processamento comum
def preprocess_text(text: str) -> list[str]:
    """
    Lowercase, strip acentos, remove não-alfanuméricos e split em tokens.
    """
    txt = text.lower()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(ch for ch in txt if unicodedata.category(ch) != "Mn")
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    return txt.split()

# 2) Definição “bruta” dos seus sinônimos como lista de listas
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

# 3) Normaliza cada grupo de sinônimos e monta token→canônico
synonyms_to_canonical: dict[str,str] = {}
for group in raw_synonyms:
    # torne cada termo “canônico” e seus sinônimos normais
    normalized = [preprocess_text(term)[0] for term in group if preprocess_text(term)]
    canonical = normalized[0]
    for tok in normalized:
        synonyms_to_canonical[tok] = canonical

# 4) Frases_com_pesos brutas (seq de “chaves canônicas”)
raw_phrases = [
    {"seq": ["gerente","atendimento","ruim"],         "peso": 0.9, "causa": "falta de contato"},
    {"seq": ["manipulado","gerente","agencia"],       "peso": 0.9, "causa": "falta de contato"},
    {"seq": ["falta","de","contato"],                 "peso": 0.7, "causa": "falta de contato"},
    {"seq": ["tentativas","contato","frustrada"],     "peso": 0.8, "causa": "falta de contato"},
]

# 5) Pré‑processa as seqüências das frases e mapeia a canônico
frases_com_pesos = []
for entry in raw_phrases:
    seq_norm: list[str] = []
    for term in entry["seq"]:
        toks = preprocess_text(term)
        if not toks: 
            continue
        # mapeia para a forma canônica (se for sinônimo)
        seq_norm.append(synonyms_to_canonical.get(toks[0], toks[0]))
    frases_com_pesos.append({
        "seq": seq_norm,
        "peso": entry["peso"],
        "causa": entry["causa"]
    })

# 6) Função de cálculo de score usando distância de tokens
def calcular_score_ponderado(
    tokens: list[str],
    frases_com_pesos: list[dict],
    synonyms_to_canonical: dict[str,str],
    window_size: int = 10
) -> tuple[float,str|None]:
    # a) indexa todas as posições de cada token canônico
    posicoes: dict[str,list[int]] = defaultdict(list)
    for i, tok in enumerate(tokens):
        canon = synonyms_to_canonical.get(tok, tok)
        posicoes[canon].append(i)

    best_score = 0.0
    best_causa = None

    # b) para cada padrão, tenta encontrar todos os termos
    for pat in frases_com_pesos:
        seq, peso, causa = pat["seq"], pat["peso"], pat["causa"]
        found = []
        for term in seq:
            lst = posicoes.get(term, [])
            if not lst:
                found = []
                break
            found.append(lst[0])  # primeira ocorrência
        if not found:
            continue

        # c) calcula distância e proximidade
        dist = max(found) - min(found)
        proximity = max(0, window_size - dist) / window_size
        score = peso * (1 + proximity)

        if score > best_score:
            best_score, best_causa = score, causa

    return best_score, best_causa

# 7) Exemplo de aplicação a um DataFrame
# -> df.texto contém o texto bruto de cada reclamação
df = pd.DataFrame({
    "texto": [
        "O gerente de suporte da agência foi ruim e não retornou.",
        "Tive falta de contato com a agência, tentativas frustradas!",
        "Produto danificado, problema na entrega."
    ]
})

# 8) Pipeline completo
df["tokens"] = df["texto"].apply(preprocess_text)
results = df["tokens"] \
    .apply(lambda toks: calcular_score_ponderado(toks, frases_com_pesos, synonyms_to_canonical, window_size=15))
df[["score","causa"]] = pd.DataFrame(results.tolist(), index=df.index)

print(df)
