from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
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

dataset = load_dataset("json", data_files=data_file, split="train")

print(dataset)
# Carica il tokenizer e il modello
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=save_directory)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=save_directory)
#tokenized_sentences = tokenizer.apply_chat_template(dataset["messages"], tokenize=True)

# x = "Hello, World!"

# output = tokenizer(x)
# print("OUTPUT: ", vars(output))
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})
#print(dataset['formatted_chat'][0])
# print(tokenized_sentences[:10])
def tokenize_function(examples):
    tokenized = tokenizer(examples["formatted_chat"], padding="max_length", truncation=True, max_length=512, return_tensors='pt')
   
    tokenized['labels'] = tokenized['input_ids']


    return tokenized 
# tokenized_data = [tokenize_function(item) for item in dataset]
# input_ids_tensor = torch.tensor(tokenized_data[0]['input_ids'])
# print(input_ids_tensor.shape)

# rimuoviamo le colonne che non vengono processate dalla classe SFTTrain
column_names = ['messages', 'formatted_chat']
tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )





# train_dataset = dataset.map(lambda x: {
#     "labels": x["input_ids"]  # Aggiungi i target (solitamente gli stessi degli input per LM)
# }, batched=True)
#print("RISULTATO: ", dataset[0])
model.resize_token_embeddings(len(tokenizer))


training_args = SFTConfig(
    max_seq_length=512,
    output_dir= output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
)
trainer = SFTTrainer(
    model = model,
    train_dataset=tokenized_datasets,
    args=training_args,
    packing=False
)

trainer.train()

