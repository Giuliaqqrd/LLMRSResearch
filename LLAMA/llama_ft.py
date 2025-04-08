import os
import torch
import gc
import itertools
import transformers
import json
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# 🔹 Percorso dello script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 🔹 Caricamento del dataset
data_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"
dataset = load_dataset("text", data_files={"train": data_file})['train']

# 🔹 Tokenizer
base_model_id = "meta-llama/Llama-2-7b"
save_directory = "/mnt/storage/huggingface/hub"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    cache_dir=save_directory
)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 Funzione per tokenizzare il dataset
def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        prompt['text'],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

# Split train/test
llamadataset = tokenized_dataset.train_test_split(test_size=0.2)

# 🔹 Possibili valori per i parametri
r_values = [8, 16]
learning_rates = [5e-5, 1e-5]
optimizers = ["paged_adamw_8bit"]
epochs_list = [3, 5]
weight_decays = [0.01]
lr_schedulers = ["linear", "cosine"]

# 🔹 Generazione di tutte le combinazioni possibili
experiments = list(itertools.product(r_values, learning_rates, optimizers, epochs_list, weight_decays, lr_schedulers))

# 🔹 Cartella base per i log (usando il nome del modello)
logs_base_dir = os.path.join(script_dir, base_model_id.replace("/", "_"))
os.makedirs(logs_base_dir, exist_ok=True)

results = []
# 🔹 Loop su tutte le configurazioni
for i, (r, lr, optim, epochs, weight_decay, lr_scheduler) in enumerate(experiments):
    lora_alpha = 2 * r  # Vincolo: lora_alpha = 2 * r

    run_name = f"run_{i+1}_r{r}_alpha{lora_alpha}_lr{lr}_opt{optim}_ep{epochs}"
    run_output_dir = os.path.join(logs_base_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # 🔹 Caricamento del modello quantizzato
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=save_directory
    )

    # 🔹 Attivazione di gradient checkpointing per ridurre la memoria
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 🔹 Configurazione di LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout= 0.1,
    )

    model = get_peft_model(model, lora_config)

    # 🔹 TrainingArguments (senza checkpoint)
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        save_strategy="no",  # ❌ Nessun checkpoint salvato
        learning_rate=lr,
        weight_decay=weight_decay,
        optim=optim,
        lr_scheduler_type=lr_scheduler,
        bf16=True,
        logging_strategy="steps",
        logging_steps=10,
        gradient_checkpointing=True,
        report_to="none"
    )

    # 🔹 Trainer
    trainer = Trainer(
        model=model,
        train_dataset=llamadataset["train"],
        eval_dataset=llamadataset["test"],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 🔹 Avvio del fine-tuning
    print(f"Starting fine-tuning: {run_name}")
     # Avvio del fine-tuning
    trainer.train()
    # Calcolo della perplexity sulla validazione
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if "eval_loss" in eval_results else float("inf")

    # Salvataggio dei log
    log_data = {
        "run_name": run_name,
        "r": r,
        "lora_alpha": lora_alpha,
        "optim": optim,
        "learning_rate": lr,
        "lr_scheduler": lr_scheduler,
        "weight_decay": weight_decay,
        "num_train_epochs": epochs,
        "train_log": trainer.state.log_history,
        "perplexity": perplexity
    }

    results.append(log_data)
    with open(os.path.join(run_output_dir, "training_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)

    # 🔹 Pulizia della memoria GPU
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Finished fine-tuning: {run_name}")

print("All experiments completed.")