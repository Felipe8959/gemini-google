from collections import defaultdict
from typing import List, Dict, Tuple, Optional

def calcular_score_com_distancia(
    tokens: List[str],
    frases_com_pesos: List[Dict[str, any]],
    root_seqs: List[List[str]],
    synonyms_to_canonical: Dict[str, str],
    window_size: int = 10
) -> Tuple[float, Optional[str]]:
    """
    Calcula o score de padrões de reclamação em relação a termos 'raiz' (e.g. 'agencia',
    'gerente agencia', 'gerente da agencia'). Para cada padrão em frases_com_pesos, mede
    a menor distância entre qualquer token do padrão e qualquer token de cada root_seq,
    aplica:
        proximity = max(0, window_size - dist) / window_size
        score = peso * (1 + proximity)
    Retorna o maior score e sua causa associada.

    tokens: lista de tokens do texto (pré‑processados)
    frases_com_pesos: [
        {'seq': ['falta','de','contato'], 'peso': 0.7, 'causa': 'falta de contato'},
        ...
    ]
    root_seqs: [
        ['agencia'],
        ['gerente','agencia'],
        ['gerente','da','agencia']
    ]
    synonyms_to_canonical: mapeia cada token/sinônimo para sua forma canônica
    window_size: número máximo de tokens para bônus de proximidade
    """
    # 1) indexa posições de cada token canônico
    posicoes: Dict[str, List[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        canon = synonyms_to_canonical.get(tok, tok)
        posicoes[canon].append(idx)

    best_score = 0.0
    best_causa: Optional[str] = None

    # 2) para cada padrão de reclamação
    for pat in frases_com_pesos:
        # mapeia a sequência para canônicos
        seq_canon = [synonyms_to_canonical.get(t, t) for t in pat['seq']]
        pat_positions = [pos for term in seq_canon for pos in posicoes.get(term, [])]
        if not pat_positions:
            continue

        # 3) compara com cada término raiz
        for root in root_seqs:
            root_canon = [synonyms_to_canonical.get(t, t) for t in root]
            root_positions = [pos for term in root_canon for pos in posicoes.get(term, [])]
            if not root_positions:
                continue

            # distância mínima entre qualquer token do padrão e da raiz
            dist = min(abs(i - j) for i in pat_positions for j in root_positions)
            proximity = max(0, window_size - dist) / window_size
            score = pat['peso'] * (1 + proximity)

            if score > best_score:
                best_score = score
                best_causa = pat['causa']

    return best_score, best_causa
