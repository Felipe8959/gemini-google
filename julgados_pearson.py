import numpy as np

# Selecionar as variáveis numéricas para análise
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Calcular correlação de Pearson da coluna 'julgado' com as outras variáveis numéricas
pearson_correlations = {}
for column in numerical_columns:
    if column != 'julgado':
        correlation = df['julgado'].corr(df[column])
        pearson_correlations[column] = correlation

# Criar DataFrame com as correlações
correlation_df = pd.DataFrame.from_dict(pearson_correlations, orient='index', columns=['Correlação com Julgamento'])
correlation_df = correlation_df.sort_values(by='Correlação com Julgamento', ascending=False)

# Calcular a correlação combinada entre pares de variáveis e a coluna 'julgado'
combined_correlation = {}
for col1 in numerical_columns:
    for col2 in numerical_columns:
        if col1 != col2 and col1 != 'julgado' and col2 != 'julgado':
            # Criar a soma das duas colunas
            combined_feature = df[col1] + df[col2]
            correlation = df['julgado'].corr(combined_feature)
            combined_correlation[f'{col1} + {col2}'] = correlation

# Criar DataFrame com as correlações combinadas
combined_correlation_df = pd.DataFrame.from_dict(combined_correlation, orient='index', columns=['Correlação com Julgamento'])
combined_correlation_df = combined_correlation_df.sort_values(by='Correlação com Julgamento', ascending=False)

# Exibir os resultados
import ace_tools as tools; tools.display_dataframe_to_user(name="Correlação de Pearson - Variáveis Individuais", dataframe=correlation_df)
tools.display_dataframe_to_user(name="Correlação de Pearson - Combinação de Variáveis", dataframe=combined_correlation_df)
