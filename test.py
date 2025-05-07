# ── 1. Instalação (ambiente ≥ Python 3.9 + GPU com 12 GB) ──────────────
!pip install -U "transformers>=4.40" datasets evaluate accelerate \
             bitsandbytes                 # (<‑‑ fp16 + LORA se quiser)

# ── 2. Carrega o dataset ------------------------------------------------
# Supondo três CSVs com colunas: text,label  (label ∈ {0,1})
from datasets import load_dataset

data_files = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv"
}
ds = load_dataset("csv", data_files=data_files)

# ── 3. Tokenização ------------------------------------------------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
        "neuralmind/bert-base-portuguese-cased", 
        do_lower_case=False)

max_len = 256             # aumente só se a maioria dos textos for longa

def tokenize(batch):
    return tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len
    )

ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

# ── 4. Modelo -----------------------------------------------------------
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            num_labels=2,
            problem_type="single_label_classification")

# (Opcional) Congelar as 4 primeiras camadas se dataset <5 k amostras:
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:4]:
    for param in layer.parameters():
        param.requires_grad = False

# ── 5. Métricas ---------------------------------------------------------
import evaluate, numpy as np
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# ── 6. Argumentos de treino --------------------------------------------
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
args = TrainingArguments(
    output_dir="bertimbau_resp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    num_train_epochs=3,            # comece com 3; aumente se não houver overfit
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,            # variação típica: 1e‑5 → 5e‑5
    weight_decay=0.01,
    warmup_ratio=0.1,              # 10 % dos steps
    lr_scheduler_type="linear",
    gradient_accumulation_steps=2, # efetivo = 32 se memória curta
    fp16=True,                     # exige GPU Ampere+ ou ROCm
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate(ds_tok["test"])
model.save_pretrained("bertimbau_resp/best")
tok.save_pretrained("bertimbau_resp/best")
