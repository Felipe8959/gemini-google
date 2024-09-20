import customtkinter as ctk
import pandas as pd
from tkinter import messagebox
from PIL import Image, ImageTk

# Configuração inicial do CustomTkinter
ctk.set_appearance_mode("Dark")  # Modo escuro
ctk.set_default_color_theme("blue")  # Tema azul padrão

# Criando a aplicação
app = ctk.CTk()
app.title("Análise de Manifestações")
app.geometry("700x500")

# Carregando a base de dados (substitua com o caminho para sua base de dados)
# Aqui estou usando um exemplo de dicionário para simular a base de dados
data = pd.read_excel('manifestacoes-ouvidoria-atualizado.xlsx')

df = pd.DataFrame(data)

# Função para buscar o protocolo e preencher os campos
def buscar_protocolo():
    protocolo = entry_protocolo.get()
    if protocolo in df["PROTOCOLO"].values:
        manifestacao = df[df["PROTOCOLO"] == protocolo].iloc[0]
        entry_descricao.delete(0, ctk.END)
        entry_descricao.insert(0, manifestacao["DESCRICAO DA MANIFESTACAO"])
        entry_palavras.delete(0, ctk.END)
        entry_palavras.insert(0, manifestacao["palavras_inapropriadas"])
        entry_resumo.delete(0, ctk.END)
        entry_resumo.insert(0, manifestacao["resumo"])
        entry_sumarizacao.delete(0, ctk.END)
        entry_sumarizacao.insert(0, manifestacao["sumarizacao"])
        atualizar_sentimento(manifestacao["sentimento"])
    else:
        messagebox.showerror("Erro", "Protocolo não encontrado!")

# Função para atualizar a carinha conforme o sentimento
def atualizar_sentimento(sentimento):
    if sentimento > 0:
        image = Image.open("smile.png")  # Adicione um ícone de carinha feliz
        image = image.resize((30, 30))
        img = ImageTk.PhotoImage(image)
        lbl_sentimento.configure(image=img)
        lbl_sentimento.image = img
    elif sentimento == 0:
        image = Image.open("neutral.png")  # Adicione um ícone de carinha neutra
        image = image.resize((30, 30))
        img = ImageTk.PhotoImage(image)
        lbl_sentimento.configure(image=img)
        lbl_sentimento.image = img
    else:
        image = Image.open("angry.png")  # Adicione um ícone de carinha triste
        image = image.resize((30, 30))
        img = ImageTk.PhotoImage(image)
        lbl_sentimento.configure(image=img)
        lbl_sentimento.image = img

# Layout da aplicação

# Campo de entrada para protocolo
label_protocolo = ctk.CTkLabel(app, text="Protocolo:")
label_protocolo.grid(row=0, column=0, padx=10, pady=10, sticky="e")

entry_protocolo = ctk.CTkEntry(app, width=200)
entry_protocolo.grid(row=0, column=1, padx=10, pady=10)

# Botão de busca
btn_buscar = ctk.CTkButton(app, text="Buscar", command=buscar_protocolo)
btn_buscar.grid(row=0, column=2, padx=10, pady=10)

# Campo de descrição
label_descricao = ctk.CTkLabel(app, text="Descrição da Manifestação:")
label_descricao.grid(row=1, column=0, padx=10, pady=10, sticky="e")

entry_descricao = ctk.CTkEntry(app, width=500)
entry_descricao.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

# Campo de palavras inapropriadas
label_palavras = ctk.CTkLabel(app, text="Palavras Inapropriadas:")
label_palavras.grid(row=2, column=0, padx=10, pady=10, sticky="e")

entry_palavras = ctk.CTkEntry(app, width=200)
entry_palavras.grid(row=2, column=1, columnspan=2, padx=10, pady=10)

# Campo de resumo
label_resumo = ctk.CTkLabel(app, text="Resumo:")
label_resumo.grid(row=3, column=0, padx=10, pady=10, sticky="e")

entry_resumo = ctk.CTkEntry(app, width=500)
entry_resumo.grid(row=3, column=1, columnspan=2, padx=10, pady=10)

# Campo de sumarização
label_sumarizacao = ctk.CTkLabel(app, text="Sumarização:")
label_sumarizacao.grid(row=4, column=0, padx=10, pady=10, sticky="e")

entry_sumarizacao = ctk.CTkEntry(app, width=500)
entry_sumarizacao.grid(row=4, column=1, columnspan=2, padx=10, pady=10)

# Label de sentimento com imagem
lbl_sentimento = ctk.CTkLabel(app, text="")
lbl_sentimento.grid(row=5, column=1, padx=10, pady=10)

# Inicializando a aplicação
app.mainloop()