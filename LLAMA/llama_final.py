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
output_dir = "/mnt/storage/optsmall"
# 🔹 Caricamento del dataset
data_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"
dataset = load_dataset("text", data_files={"train": data_file})['train']


base_model_id = "meta-llama/Meta-Llama-3-8B"
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
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout= 0.1,
)

model = get_peft_model(model, lora_config)

# 🔹 TrainingArguments (senza checkpoint)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_strategy="no",  # ❌ Nessun checkpoint salvato
    learning_rate=5e-5,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    lr_scheduler_type="constant",
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

trainer.train()





final_model_dir = os.path.join(output_dir, base_model_id.replace("/", "_"))
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\n===== All runs completed! Model saved. =====\n")
