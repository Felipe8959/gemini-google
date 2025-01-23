import pandas as pd
import joblib

# 1. Carregar a pipeline salva
pipeline_produto = joblib.load('pipeline_produto.pkl')

# 2. Ler o novo DataFrame que você quer classificar
df_novo = pd.read_csv('dados_novos.csv', sep=';', encoding='utf-8')

# 3. Criar (novamente) a coluna de texto_unificado da mesma forma
df_novo['texto_completo'] = (
    df_novo['DESCR FAMILIA'].fillna('') + ' ' +
    df_novo['DESCR PRODUTO'].fillna('') + ' ' +
    df_novo['DESCR ASSUNTO'].fillna('')
)

# 4. Prever usando a pipeline
X_novo = df_novo['texto_completo']
y_pred_novo = pipeline_produto.predict(X_novo)

# 5. Incluir o resultado de previsão no dataframe se quiser analisar
df_novo['PRODUTO_FEBRABAN_PRED'] = y_pred_novo

# 6. Visualizar
print(df_novo[['texto_completo','PRODUTO_FEBRABAN_PRED']].head())
