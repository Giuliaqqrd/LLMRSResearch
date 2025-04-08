import os
import json
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Features, Value

# CONFIGURAZIONE BASE
save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/optsmall"
base_model_id = "facebook/opt-350m"
data_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"

# Carica dataset
features = Features({"text": Value("string")})
dataset = load_dataset("text", data_files={"train": data_file}, split="train")

tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=save_directory)

# Funzione di tokenizzazione
def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split train/test
optdataset = tokenized_datasets.train_test_split(test_size=0.2)

# Definizione delle combinazioni di iperparametri
lr = 5e-5
lr_scheduler = "cosine"
weight_decay= 0.001
num_epochs = 10


# Carica il modello una sola volta
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", cache_dir=save_directory)
model.config.use_cache = False  # Necessario per il training efficiente



training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=lr,
    lr_scheduler_type= lr_scheduler,
    weight_decay=weight_decay,
    per_device_train_batch_size=32,
    num_train_epochs=num_epochs,
    logging_steps=10,
    logging_first_step=True,
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=optdataset["train"],
    eval_dataset=optdataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Avvio del fine-tuning
trainer.train()
# Calcolo della perplexity sulla validazione
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
perplexity = math.exp(eval_loss) if "eval_loss" in eval_results else float("inf")



# Salvataggio finale del modello
final_model_dir = os.path.join(output_dir, base_model_id.replace("/", "_"))
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\n===== All runs completed! Model saved. =====\n")