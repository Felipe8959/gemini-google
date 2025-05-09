Fine‑tuning BERTimbau para classificar respostas completas (1) vs. incompletas (0)

-----------------------------------------------------------------------------

Pré‑requisitos (execute uma única vez, p.ex. num notebook Colab, VSCode, etc.)

!pip install -q transformers datasets evaluate scikit-learn accelerate

import pandas as pd from datasets import Dataset from transformers import ( AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, ) import evaluate import numpy as np from sklearn.model_selection import train_test_split

-----------------------------------------------------------------------------

1. Carregar o DataFrame data

-----------------------------------------------------------------------------

Supondo que já exista na sessão (colunas "text_full" e "label")

Caso esteja em CSV/parquet, descomente:

data = pd.read_csv("respostas_clientes.csv")

assert {"text_full", "label"}.issubset(data.columns), "Colunas obrigatórias não encontradas."

df = data[["text_full", "label"]].rename(columns={"text_full": "text", "label": "label"})

-----------------------------------------------------------------------------

2. Dividir em treino e validação estratificados

-----------------------------------------------------------------------------

train_df, val_df = train_test_split( df, test_size=0.2, stratify=df["label"], random_state=42 )

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)) val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

-----------------------------------------------------------------------------

3. Tokenizar com BERTimbau (cased)

-----------------------------------------------------------------------------

model_id = "neuralmind/bert-base-portuguese-cased"

print("Baixando tokenizer & modelo…") tokenizer = AutoTokenizer.from_pretrained(model_id)

MAX_LENGTH = 256  # truncate/pad a 256 tokens (≈ 512 é o limite de BERT)

def preprocess(batch): return tokenizer( batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH, )

train_ds = train_ds.map(preprocess, batched=True, remove_columns=["text"]) val_ds = val_ds.map(preprocess, batched=True, remove_columns=["text"])

-----------------------------------------------------------------------------

4. Data collator e modelo

-----------------------------------------------------------------------------

collator = DataCollatorWithPadding(tokenizer) model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

-----------------------------------------------------------------------------

5. Métricas

-----------------------------------------------------------------------------

accuracy = evaluate.load("accuracy") f1 = evaluate.load("f1")

def compute_metrics(eval_pred): logits, labels = eval_pred preds = np.argmax(logits, axis=-1) acc = accuracy.compute(predictions=preds, references=labels)["accuracy"] f1_macro = f1.compute(predictions=preds, references=labels, average="macro")["f1"] return {"accuracy": acc, "f1": f1_macro}

-----------------------------------------------------------------------------

6. Parâmetros de treinamento

-----------------------------------------------------------------------------

args = TrainingArguments( output_dir="bertimbau-classifier", evaluation_strategy="epoch", save_strategy="epoch", learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=3, weight_decay=0.01, logging_steps=50, load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, report_to="none",  # desabilita W&B se não usado )

trainer = Trainer( model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics, )

-----------------------------------------------------------------------------

7. Treinar

-----------------------------------------------------------------------------

print("\n==== Iniciando treino ====") trainer.train()

-----------------------------------------------------------------------------

8. Avaliação final e salvamento

-----------------------------------------------------------------------------

print("\n==== Avaliação final ====") metrics = trainer.evaluate() print(metrics)

trainer.save_model("bertimbau-classifier/best") print("\nModelo salvo em bertimbau-classifier/best")

-----------------------------------------------------------------------------

9. Função de inferência prática

-----------------------------------------------------------------------------

import torch

def predict(text: str): """Retorna tupla (label_predito, probabilidade).""" model.eval() with torch.no_grad(): inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH) outputs = model(**inputs) probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy() label = int(np.argmax(probs)) return label, float(probs[label])

Exemplo rápido

if name == "main": exemplo = "Cliente não atendeu após 3 tentativas [SEP] Precisamos de retorno para concluir a solicitação." lbl, p = predict(exemplo) print(f"Label predito: {lbl} | confiança: {p:.3f}")

