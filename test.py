# Criar um DataFrame com a volumetria por hora
volumetria_por_hora = df['Hora Atualização'].value_counts().reset_index()
volumetria_por_hora.columns = ['Hora Atualização', 'Volumetria']

# Garantir que a coluna 'Hora Atualização' seja numérica
volumetria_por_hora['Hora Atualização'] = pd.to_numeric(volumetria_por_hora['Hora Atualização'], errors='coerce')

# Aplicar o filtro para considerar apenas entre 9 e 18 e volumetria >= 50
volumetria_por_hora = volumetria_por_hora[
    (volumetria_por_hora['Hora Atualização'] >= 9) &
    (volumetria_por_hora['Hora Atualização'] <= 18) &
    (volumetria_por_hora['Volumetria'] >= 50)
]

# Exibir o resultado
print(volumetria_por_hora)
