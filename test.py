import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Exemplo de dados
dados = {
    'data': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'volumetria': [100, 150, 200, 180, 220, 170, 160, 210, 230, 250, 240, 260],
    'Interno': [20, 25, 22, 30, 28, 26, 27, 29, 31, 32, 30, 28],
    'Externo': [80, 75, 78, 70, 72, 74, 73, 71, 69, 68, 70, 72]
}
df = pd.DataFrame(dados)

# Configuração do gráfico
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotar a volumetria como barras
ax1.bar(df['data'], df['volumetria'], color='skyblue', label='Volumetria', width=20)
ax1.set_xlabel('Mês')
ax1.set_ylabel('Volumetria')

# Formatar o eixo X para exibir apenas os meses (abreviados)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())

# Criar um segundo eixo para os percentuais
ax2 = ax1.twinx()
ax2.plot(df['data'], df['Interno'], color='green', marker='o', label='Interno')
ax2.plot(df['data'], df['Externo'], color='red', marker='o', label='Externo')
ax2.set_ylabel('Percentual (%)')

# Unir legendas dos dois eixos
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Gráfico Combo: Volumetria e Percentuais Interno/Externo')
plt.tight_layout()
plt.show()
