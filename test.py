import os
from datetime import datetime

def listar_ultimas_interacoes(pasta):
    try:
        arquivos = os.listdir(pasta)
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(pasta, arquivo)
            if os.path.isfile(caminho_arquivo):
                ultima_modificacao = os.path.getmtime(caminho_arquivo)
                data_formatada = datetime.fromtimestamp(ultima_modificacao).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Arquivo: {arquivo} - Última Interação: {data_formatada}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Exemplo de uso:
pasta_especifica = '/caminho/para/sua/pasta'
listar_ultimas_interacoes(pasta_especifica)
