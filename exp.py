import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)
from safetensors.torch import load_file
from accelerate import Accelerator
from transformers import get_scheduler

# === CONFIGURAZIONE ===
save_directory = "/mnt/storage/huggingface/hub"  # Directory per il salvataggio del modello/tokenizer
output_dir = "/mnt/storage/huggingface/hub"          # Directory per i checkpoint
data_file = "data/output_data_qwen.jsonl"        # File JSON con i dati

# === CARICAMENTO DEL DATASET ===
dataset = load_dataset("json", data_files=data_file, split="train")

# Preprocessing per il template delle chat
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", cache_dir=save_directory, from_safetensors=True)
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta", 
    cache_dir=save_directory, 
    device_map="auto", 
    use_cache=False
)

# Adatta il dataset al formato del modello
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(
    x["messages"], tokenize=False, add_generation_prompt=False)})

# Funzione di tokenizzazione
def tokenize_function(examples):
    tokenized = tokenizer(examples["formatted_chat"], 
                          padding="max_length", 
                          truncation=True, 
                          max_length=512, 
                          return_tensors='pt')
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Tokenizza il dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Suddivisione del dataset in train, validation e test
train_test = tokenized_datasets.train_test_split(test_size=0.3, seed=42)
validation_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
dataset_dict = DatasetDict({
    "train": train_test["train"],
    "validation": validation_test["train"],
    "test": validation_test["test"]
})

# Rimuovi colonne non necessarie
dataset_dict = dataset_dict.remove_columns(["formatted_chat", "messages"])

# === CREAZIONE DEI DATALOADER ===
def create_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: {
            "input_ids": torch.tensor([item["input_ids"] for item in x], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in x], dtype=torch.long),
            "labels": torch.tensor([item["labels"] for item in x], dtype=torch.long),
        }
    )

train_dataloader = create_dataloader(dataset_dict["train"], batch_size=1, shuffle=True)
eval_dataloader = create_dataloader(dataset_dict["validation"], batch_size=1)

# === ACCELERATE ===
accelerator = Accelerator(project_dir=output_dir)

# Preparazione dei componenti con Accelerate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=100
)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
accelerator.register_for_checkpointing(lr_scheduler)

# === RIPRESA CHECKPOINT ===
if os.path.isdir(output_dir):
    accelerator.print("Checkpoint trovato, caricamento stato precedente...")
    accelerator.load_state(output_dir)

# === CICLO DI TRAINING ===
num_epochs = 1  # Imposta il numero di epoche
model.train()
for epoch in range(num_epochs):
    accelerator.print(f"Epoch {epoch + 1}/{num_epochs}")
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        # Logging della loss
        if step % 5 == 0:
            accelerator.print(f"Step {step}, Loss: {loss.item()}")

        # Salvataggio periodico dello stato
        if step % 10 == 0:
            accelerator.save_state()

    # Validazione dopo ogni epoca
    model.eval()
    total_eval_loss = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            total_eval_loss += outputs.loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    accelerator.print(f"Validation Loss: {avg_eval_loss}")

    # Torna alla modalità di training
    model.train()

# === SALVATAGGIO FINALE ===
accelerator.save_state()
accelerator.print("Addestramento completato e checkpoint salvato.")
