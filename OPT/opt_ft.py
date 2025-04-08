import os
import json
import math
import itertools
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, Features, Value

# CONFIGURAZIONE BASE
save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/optsmall"
base_model_id = "facebook/opt-1.3b"
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
param_grid = {
    "learning_rate": [2e-5],  # Learning rate sensati
    "lr_scheduler": ["linear", "cosine"],  # Scheduler ragionevoli
    "weight_decay": [0.01, 0.001],  # Regolarizzazione leggera
    "num_train_epochs": [3, 10]  # Numero di epoche
}


# Genera tutte le combinazioni di iperparametri
param_combinations = list(itertools.product(
    param_grid["learning_rate"],
    param_grid["lr_scheduler"],
    param_grid["weight_decay"],
    param_grid["num_train_epochs"]
))

# Lista per salvare i risultati
results = []

# LOOP SU OGNI CONFIGURAZIONE DI FINETUNING
for i, (lr, scheduler_type, weight_decay, num_epochs) in enumerate(param_combinations):
    run_name = f"run_{i+1}_facebookopt1.3_lr{lr}_sched{scheduler_type}_wd{weight_decay}_epochs{num_epochs}"
    run_output_dir = os.path.join(os.getcwd(), run_name)
    checkpoint_output_dir = os.path.join(output_dir, "facebookopt1.3", run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"\n===== Starting {run_name} =====\n")

    # Carica il modello ogni volta per evitare interferenze tra esperimenti
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", cache_dir=save_directory)

    training_args = TrainingArguments(
        output_dir=checkpoint_output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        warmup_ratio=0.1,
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
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
    )

    model.config.use_cache = False  # Necessario per il training efficiente

    # Avvio del fine-tuning
    trainer.train()
    # Calcolo della perplexity sulla validazione
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if "eval_loss" in eval_results else float("inf")

    # Salvataggio dei log
    log_data = {
        "run_name": run_name,
        "learning_rate": lr,
        "lr_scheduler": scheduler_type,
        "weight_decay": weight_decay,
        "num_train_epochs": num_epochs,
        "train_log": trainer.state.log_history,
        "perplexity": perplexity
    }

    results.append(log_data)
    with open(os.path.join(run_output_dir, "training_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Completed {run_name}")

print("\n===== All runs completed! =====\n")
