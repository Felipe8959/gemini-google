import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Carregar os dados
file_path = "/mnt/data/seu_arquivo.csv"  # Substitua pelo nome correto do arquivo
data = pd.read_csv(file_path, encoding="latin1")  # Tente ajustar a codificação conforme necessário

# Supondo que as colunas relevantes sejam:
# 'ANALISE_RESPOSTA_I' (rótulo real) e 'probabilities' (probabilidade predita da classe positiva)

# Separando as probabilidades preditas para cada classe
probs_class_0 = data[data['ANALISE_RESPOSTA_I'] == 0]['probabilities']
probs_class_1 = data[data['ANALISE_RESPOSTA_I'] == 1]['probabilities']

# Ordenando os valores para calcular a CDF
probs_class_0_sorted = np.sort(probs_class_0)
probs_class_1_sorted = np.sort(probs_class_1)

# Criando as funções de distribuição acumulada (CDF)
cdf_class_0 = np.arange(1, len(probs_class_0_sorted) + 1) / len(probs_class_0_sorted)
cdf_class_1 = np.arange(1, len(probs_class_1_sorted) + 1) / len(probs_class_1_sorted)

# Calculando o KS Statistic
ks_stat, _ = ks_2samp(probs_class_0, probs_class_1)

# Plotando o gráfico de KS
plt.figure(figsize=(8, 6))
plt.plot(probs_class_0_sorted, cdf_class_0, label="Classe 0", color="blue")
plt.plot(probs_class_1_sorted, cdf_class_1, label="Classe 1", color="red")

# Destacando o ponto de máxima separação
idx_max_sep = np.argmax(np.abs(cdf_class_0 - cdf_class_1))
plt.vlines(probs_class_0_sorted[idx_max_sep], cdf_class_0[idx_max_sep], cdf_class_1[idx_max_sep],
           colors="black", linestyles="dashed", label=f"KS = {ks_stat:.3f}")

# Configurações do gráfico
plt.xlabel("Probabilidade Predita")
plt.ylabel("Função de Distribuição Acumulada (CDF)")
plt.title("Curvas KS - Separação entre Classes")
plt.legend()
plt.grid()

# Exibir o gráfico
plt.show()
