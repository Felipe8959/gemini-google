#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Hugging Face Transformers
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Exemplo de dados simulados
# Substitua por seus dados reais
textos_train = [
    "Essa resposta está completa e bem detalhada.",
    "Não há informação suficiente aqui.",
    "O texto cobre todos os pontos necessários.",
    "Faltam vários detalhes importantes."
]
labels_train = [1, 0, 1, 0]  # 1 = completa, 0 = incompleta

textos_val = [
    "A resposta cobre parcialmente o que precisa.",
    "Está muito bem explicada e com detalhes."
]
labels_val = [0, 1]

####################################################################
# 1. Definindo um Dataset personalizado para tokenizar e armazenar
####################################################################
class RespostasDataset(Dataset):
    def __init__(self, textos, labels, tokenizer, max_length=128):
        self.textos = textos
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        label = self.labels[idx]

        # Tokeniza o texto
        tokens = self.tokenizer(
            texto,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Cria um dicionário com input_ids, attention_mask e label
        item = {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        return item


####################################################################
# 2. Função de Métricas (Accuracy, Precision, Recall, F1)
####################################################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

####################################################################
# 3. Carregando Tokenizer e Modelo BERT
####################################################################
# Escolha um modelo pré-treinado apropriado.
# Exemplos: 'bert-base-uncased' (Inglês), 'neuralmind/bert-base-portuguese-cased' (Português), etc.
# Aqui, exemplo em português:
model_name = "neuralmind/bert-base-portuguese-cased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Classificação binária
)

####################################################################
# 4. Preparando os Datasets de Treino e Validação
####################################################################
train_dataset = RespostasDataset(textos_train, labels_train, tokenizer)
val_dataset = RespostasDataset(textos_val, labels_val, tokenizer)

####################################################################
# 5. Configurando Parâmetros de Treino
####################################################################
training_args = TrainingArguments(
    output_dir='./modelo_bert_respostas',   # Diretório para salvar checkpoints
    num_train_epochs=3,                     # Ajuste conforme a necessidade
    per_device_train_batch_size=4,          # Ajuste para caber na GPU/CPU
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",           # Avalia ao final de cada época
    logging_steps=10,                       # Intervalos de logging
    save_steps=50,                          # Salva checkpoints a cada X steps
    load_best_model_at_end=True,            # Carrega o melhor modelo ao final
    metric_for_best_model="accuracy",       # Qual métrica usar para "melhor modelo"
    greater_is_better=True                  # Se a métrica maior é melhor
)

####################################################################
# 6. Montando o Trainer
####################################################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

####################################################################
# 7. Executando o Treinamento
####################################################################
trainer.train()

####################################################################
# 8. Avaliação Final no Conjunto de Validação
####################################################################
eval_results = trainer.evaluate()
print("Resultados de avaliação:", eval_results)

####################################################################
# 9. Salvando o Modelo Treinado
####################################################################
trainer.save_model("./modelo_classificador_completo_incompleto")

####################################################################
# 10. Exemplo de Inferência em Novos Textos
####################################################################
novos_textos = [
    "A explicação está extremamente detalhada e não falta nada.",
    "O autor não desenvolveu o assunto o suficiente."
]

# Tokenizar e prever
for texto in novos_textos:
    tokens_novo = tokenizer(
        texto,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        output = model(**tokens_novo)
        logits = output.logits
        pred = torch.argmax(logits, dim=1).item()  # 0 ou 1
        classe = "Completa" if pred == 1 else "Incompleta"
    print(f"Texto: {texto}")
    print(f"Classe prevista: {classe}")
    print("-"*50)
