from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer, setup_chat_format
import json
from datasets import Dataset, load_dataset
import torch
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# directory di salvataggio e file di input e output
data_file = "data/output_data_qwen.jsonl"
save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/tmp/qwen_finetuned"

#caricamento del dataset
dataset = load_dataset("json", data_files=data_file, split="train")

# caricamento del tokenizer e del modello 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=save_directory)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=save_directory)

dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})
def tokenize_function(examples):
    tokenized = tokenizer(examples["formatted_chat"], padding="max_length", truncation=True, max_length=512, return_tensors='pt')
    return tokenized 

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#print(tokenized_datasets[0])

training_args = TrainingArguments(
    output_dir= output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    label_names=["input_ids"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()