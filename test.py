from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Criar sessão Spark
spark = SparkSession.builder.appName("FiltrarErrosIngestao").getOrCreate()

# Exemplo de DataFrame com respostas do analista
data = [
    (1, "Olá, voc está bem?"), 
    (2, "A ag ncia já recebeu o pagamento"), 
    (3, "O alô foi recebido"), 
    (4, "Estamos verificando a situação da agência"),
    (5, "Nenhum problema identificado aqui"),
]
df = spark.createDataFrame(data, ["id", "resposta"])

# Lista de palavras suspeitas (tokens com erro)
palavras_suspeitas = ["voc", "ag ncia", "al"]

# Criar regex para encontrar palavras inteiras
regex_pattern = r"\b(" + "|".join(palavras_suspeitas) + r")\b"

# Filtrar apenas registros que contenham palavras da lista
df_filtrado = df.filter(col("resposta").rlike(regex_pattern))

# Mostrar os registros que contêm palavras com erro
df_filtrado.show(truncate=False)
