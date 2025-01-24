from pyspark.sql.functions import broadcast

df_tdoc = spark.table("pr_platfun.psdc_ing.docto_pessoa_unic_hive")
df_tuf = spark.table("pr_platfun.psdc_ing.ender_pessoa")

df_tdoc = broadcast(df_tdoc)  # Aplica broadcast se o DF for pequeno

df_base = df_tuf.join(df_tdoc, df_tuf.cclub == df_tdoc.cclub, "left_outer") \
    .filter(df_tuf.csgl_uf.isNotNull()) \
    .filter(df_tuf.csgl_uf != '') \
    .selectExpr("UPPER(csgl_uf) AS UF_CLIENTE",
                "CONCAT(LPAD(CAST(cpf_cnpj_nro AS STRING), 9, '0'), " +
                "LPAD(CAST(cpf_cnpj_fil AS STRING), 4, '0'), " +
                "LPAD(CAST(cpf_cnpj_ctr AS STRING), 2, '0')) AS CPF_CNPJ_COMPLETO") \
    .distinct() \
    .limit(100)

df_base.show()
