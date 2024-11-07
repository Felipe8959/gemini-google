import streamlit as st

# Título do aplicativo
st.title("Meu Primeiro Aplicativo Streamlit")

# Subtítulo
st.subheader("Introdução ao Streamlit")

# Texto
st.write("Este é um exemplo simples de aplicação web usando Streamlit!")

# Input do usuário
nome = st.text_input("Qual é o seu nome?")

# Botão
if st.button("Enviar"):
    st.write(f"Olá, {nome}! Bem-vindo ao Streamlit.")
