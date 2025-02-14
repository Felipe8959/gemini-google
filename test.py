# Supondo que cada protocolo seja um dicionário com as chaves:
# 'numero_protocolo', 'propensao', 'data_abertura', 'data_vencimento'

# Inicializa um dicionário para armazenar a contagem de respostas por data
contagem_respostas = {}  # Ex: {'2025-03-01': 3, '2025-03-02': 5, ...}

# Função auxiliar para gerar a lista de datas entre duas datas (inclusive)
def gerar_datas(inicio, fim):
    datas = []
    data_atual = inicio
    while data_atual <= fim:
        datas.append(data_atual)
        data_atual += timedelta(days=1)
    return datas

# Ordena protocolos pela propensão (maior prioridade primeiro)
protocolos_ordenados = sorted(protocolos, key=lambda x: x['propensao'], reverse=True)

for protocolo in protocolos_ordenados:
    # Define a janela de agendamento
    inicio = protocolo['data_abertura'] + timedelta(days=5)
    fim = protocolo['data_vencimento']
    
    # Gera a lista de datas possíveis
    datas_possiveis = gerar_datas(inicio, fim)
    
    # Seleciona a data com menor contagem de agendamentos
    data_escolhida = None
    menor_contagem = float('inf')
    
    for data in datas_possiveis:
        # Converte data para string ou outro formato consistente, se necessário
        chave = data.strftime("%Y-%m-%d")
        contagem = contagem_respostas.get(chave, 0)
        
        if contagem < menor_contagem:
            menor_contagem = contagem
            data_escolhida = data
        # Em caso de empate, pode-se optar pelo dia mais próximo do início (já que percorremos em ordem)
    
    # Atribui a data escolhida ao protocolo (ex: adicionando uma nova chave)
    protocolo['data_resposta'] = data_escolhida
    
    # Atualiza a contagem para a data escolhida
    chave = data_escolhida.strftime("%Y-%m-%d")
    contagem_respostas[chave] = contagem_respostas.get(chave, 0) + 1
