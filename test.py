from collections import defaultdict

def calcular_score_ponderado(
    texto: str,
    frases_com_pesos: dict[str, float],
    sinonimo_para_chave: dict[str, str],
    window_size: int = 10
) -> float:
    """
    Calcula um score [0,1] para `texto`, somando para cada padrão:
      peso * (1 + proximidade),
    onde proximidade ∈ [0,1] é inversamente proporcional à distância
    entre o primeiro e o último termo do padrão no texto.

    Parâmetros
    ----------
    texto : str
        Texto já normalizado (minusculas, sem acentos, etc.).
    frases_com_pesos : dict
        Mapeia cada "frase" (sequência de tokens) para seu peso.
        Ex.: {'gerente atendimento ruim': 0.9, ...}
    sinonimo_para_chave : dict
        Mapeia cada sinônimo para sua chave canônica.
        Ex.: {'gerencia':'gerente','suporte':'atendimento',...}
    window_size : int
        Máximo de tokens entre primeiro e último termo para
        ainda gerar proximidade > 0.

    Retorna
    -------
    score_normalizado : float
        score agregado dividido pelo máximo possível (2 * soma dos pesos),
        garantindo resultado em [0,1].
    """

    # 1) indexa posições de cada chave no texto
    tokens_texto = normalizar(texto)  # por ex. split em whitespace
    posicoes = defaultdict(list)
    for idx, tok in enumerate(tokens_texto):
        if tok in sinonimo_para_chave:
            chave = sinonimo_para_chave[tok]
            posicoes[chave].append(idx)

    # 2) para cada padrão, calcula seu score ponderado
    total_raw = 0.0
    peso_total = sum(frases_com_pesos.values())

    for frase, peso in frases_com_pesos.items():
        # tokens da "frase" e mapeamento para chaves
        seq = [sinonimo_para_chave[t] if t in sinonimo_para_chave else t
               for t in normalizar(frase)]
        # coleta primeira ocorrência de cada termo, se existir
        found_positions = []
        for term in seq:
            if posicoes.get(term):
                found_positions.append(posicoes[term][0])
            else:
                found_positions = []
                break
        if not found_positions:
            continue

        # distância bruta e proximidade normalizada
        dist = max(found_positions) - min(found_positions)
        proximity = max(0, window_size - dist) / window_size

        # score deste padrão
        total_raw += peso * (1 + proximity)

    # 3) normaliza dividindo pelo máximo possível: 2 * peso_total
    if peso_total == 0:
        return 0.0
    return total_raw / (2 * peso_total)
