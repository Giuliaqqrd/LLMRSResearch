from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
import torch
import pandas as pd
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader
import math
import json
import os
import gc

save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/lama8instr"

PAD_TOKEN = "<|pad|>"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NEW_MODEL = "Llama-3-8B-RS2"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast = True, cache_dir=save_directory)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = quantization_config,
    device_map = "auto",
    cache_dir=save_directory,
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

print(model.config)
print("Adding pad: ",tokenizer.pad_token, tokenizer.pad_token_id)
tokenizer.convert_tokens_to_ids(PAD_TOKEN)


# Download data
data_file = "data/tune2/tuning_dataset_final.jsonl"
dataset = load_dataset("json", data_files=data_file)

# print(dataset)
# print(dataset["train"][:5])



# Funzione per applicare il template ai messaggi
def apply_template_to_messages(example):
    # Applica apply_chat_template alla lista di messaggi dell'esempio
    formatted_chat = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    # Restituisci un dizionario con la nuova chiave 'formatted_chat'
    return {"text": formatted_chat}

# Usa map per applicare la funzione a tutti gli elementi del dataset
dataset = dataset.map(apply_template_to_messages)

# Estrai la colonna 'formatted_chat' dal dataset
formatted_chat_data = [item["text"] for item in dataset["train"]]

# Creare un DataFrame con la colonna 'formatted_chat'
df = pd.DataFrame(formatted_chat_data, columns=["text"])

# Mostra il DataFrame
 # Mostra i primi 5 risultati

def count_tokens(row):
    return len(tokenizer(
            row["text"],
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"])

#df["token_count"] = df.apply(count_tokens, axis=1)

""" print(df.head()) 
print(df.text.iloc[0])
 """


# Creazione delle partizioni per il training

#df = df[df.token_count < 512]
#df = df.sample(6000)
#print(df.shape)

# Divisione del DataFrame in train (80%), temp (20%)
train, temp = train_test_split(df, test_size=0.2)

# Suddivisione del set temporaneo in val (80% di temp) e test (20% di temp)
val, test = train_test_split(temp, test_size=0.25)  # 0.25 di temp è 20% di df

train.to_json("data/tune2/train_final.jsonl", orient="records", lines=True)
val.to_json("data/tune2/val_final.jsonl", orient="records", lines=True)
test.to_json("data/tune2/test_final.jsonl", orient="records", lines=True) 

train_set = "data/tune2/train_final.jsonl"
validation_set = "data/tune2/val_final.jsonl"
test_set = "data/tune2/test_final.jsonl"

dataset_dict = load_dataset(
    "json",
    data_files={"train": train_set, "validation": validation_set, "test": test_set}
)

#print(dataset_dict)

# Training

response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

examples = [dataset["train"][0]["text"]]
encodings = [tokenizer(e) for e in examples]

dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)

batch = next(iter(dataloader))
""" print(batch.keys())
print(batch["labels"]) """


# LoRA Setup
r=8
lora_alpha = 16

lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# Salvataggio dei log
run_name="run_8"
lr= 1.5e-5
lr_scheduler="linear"
num_train_epochs=3


sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    max_seq_length=512,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=lr,
    fp16=True,
    save_strategy="no",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type=lr_scheduler,
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

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
    "learning_rate": lr,
    "lr_scheduler": lr_scheduler,
    "num_train_epochs": num_train_epochs,
    "train_log": trainer.state.log_history,
    "perplexity": perplexity
}

# Definire il percorso della directory
log_dir = "/home/ubuntu/projects/LLMRSResearch/LLAMA"

# Creare il percorso del file di log con il prefisso run_name
log_file_path = os.path.join(log_dir, f"{run_name}_log_data.json")

# Salvataggio in formato JSON
with open(log_file_path, "w") as json_file:
    json.dump(log_data, json_file, indent=4)

print(f"Log data saved to {log_file_path}")
trainer.save_model(output_dir)

del model
torch.cuda.empty_cache()
gc.collect()