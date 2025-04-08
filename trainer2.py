import torch
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict 
from transformers.trainer_utils import get_last_checkpoint
import os
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(torch.cuda.device_count())

""" metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels) """

# Dove salvare modello
save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/checkpoints"
# Caricamento del dataset
data_file = "data/output_data_qwen.jsonl"
dataset = load_dataset("json", data_files=data_file, split="train")
#print(dataset['messages'])

""" # Suddividere il dataset in train e temp (temporary, per ricavare validation e test)
train_test = dataset.train_test_split(test_size=0.3, seed=42)

# Suddividere temp in validation e test
validation_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
 """
# Creare il DatasetDict
""" dataset_dict = DatasetDict({
    "train": train_test["train"],
    "validation": validation_test["train"],
    "test": validation_test["test"]
})

 """
# Visualizzare il risultato
#rint(dataset_dict)

# Caricamento modello e tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", cache_dir=save_directory)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", cache_dir=save_directory, device_map="auto",  use_cache=False)


# Adattare il dataset al formato richiesto per il processamento da parte del modello
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})
#print(dataset)
def tokenize_function(examples):
    tokenized = tokenizer(examples["formatted_chat"], padding="max_length", truncation=True, max_length=512, return_tensors='pt')
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized 


#print(dataset['formatted_chat'][0])

# Visualizzare un esempio da ciascuna partizione
#for split in tokenized_datasets:
#    print(f"Example from {split}:")
#    print(tokenized_datasets[split][0])
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
""" samples = tokenized_datasets["train"].select(range(16))
samples_eval = tokenized_datasets["train"].select(range(16,32)) """

# Mantieni solo le chiavi pertinenti (ad esempio: input_ids, attention_mask, ecc.)
# Rimuovi eventuali colonne extra che non sono necessarie (personalizza in base alle tue colonne)
#samples = samples.remove_columns(["formatted_chat", "messages"])
#dataset = dataset.remove_columns(["messages"])
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Calcola la lunghezza di ogni sequenza di input_ids
""" sequence_lengths = [len(x) for x in samples["input_ids"]]

# Stampa le lunghezze delle sequenze
print(sequence_lengths) """

#batch_train = data_collator(samples)
#batch_eval = data_collator(samples_eval)
#print({k: v.shape for k, v in batch.items()})

train_test = tokenized_datasets.train_test_split(test_size=0.3, seed=42)
print("Train-test: ", train_test)
validation_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
dataset_dict = DatasetDict({
    "train": train_test["train"],
    "validation": validation_test["train"],
    "test": validation_test["test"]
})

print("dataset con dizionario: ", dataset_dict)
# Stampa dell'intero dataset per ispezionare la struttura
# Stampa delle prime 5 righe del dataset 'train'
""" for i in range(5):
    print(dataset_dict['train'][i]) """

dataset_dict = dataset_dict.remove_columns(["formatted_chat", "messages"])
print("dataset con dizionario: ", dataset_dict)

""" for batch in dataset_dict["train"][0]["formatted_chat"].with_format("torch"):
    print(batch)
    print("Input IDs shape:", batch['input_ids'].shape)
    print("Attention mask shape:", batch['attention_mask'].shape) """

""" for i in range(5):
    print("Trova tensori: ", dataset_dict['train'][i]['input_ids'].shape) """


for i in range(len(dataset_dict['train'])):
    # Converto input_ids, attention_mask e labels in tensori
    input_ids = torch.tensor(dataset_dict['train'][i]['input_ids'])
    attention_mask = torch.tensor(dataset_dict['train'][i]['attention_mask'])
    labels = torch.tensor(dataset_dict['train'][i]['labels'])

for i in range(len(dataset_dict['validation'])):
    # Converto input_ids, attention_mask e labels in tensori
    input_ids = torch.tensor(dataset_dict['validation'][i]['input_ids'])
    attention_mask = torch.tensor(dataset_dict['validation'][i]['attention_mask'])

# Controlla se esiste un checkpoint
#last_checkpoint = None
print("Dataset convertito: ", dataset_dict['train'][0])
print(type(dataset_dict['train'][0]['input_ids']))  # Dovrebbe essere un elenco o un array NumPy
print(type(dataset_dict['train'][0]['input_ids'][0]))
print(type(dataset_dict['train'][0]['input_ids']))  # Dovrebbe stampare <class 'torch.Tensor'>


# training_args = TrainingArguments(
#      output_dir=output_dir,              
#      overwrite_output_dir=True, 
#      max_steps=100,  
#      optim="adamw_torch",      
#      per_device_train_batch_size=1, 
#      per_device_eval_batch_size=1,
#      gradient_accumulation_steps=5,
#      gradient_checkpointing=True,      
#      logging_strategy="steps",
#      logging_steps=1,
#      learning_rate=1e-5,
#      save_steps=5,
#      bf16=True,
#      save_total_limit=2, 
#      weight_decay=0.01,
#      max_grad_norm=1.0
  
# )

# training_args.use_cache = False
# model.config.use_cache = False
#  # Initialize the Trainer
# trainer = Trainer(
#      model=model,                         # The model to train
#      args=training_args,                  # Training arguments
#      train_dataset=dataset_dict["train"],  # Training dataset
#      eval_dataset=dataset_dict["validation"],  # Validation dataset
#      processing_class=tokenizer,                 # Tokenizer
             
# )

# # # Start training
# #trainer.train(resume_from_checkpoint=last_checkpoint)
# if os.path.isdir(output_dir):
#     last_checkpoint = get_last_checkpoint(output_dir)
#     if last_checkpoint is None:
#         print("Nessun checkpoint trovato. L'addestramento partirà da zero.")
#         trainer.train()
#     else:
#         print(f"Riprendere l'addestramento da: {last_checkpoint}")
#         trainer.train(resume_from_checkpoint=last_checkpoint)
# # Save the final model
# trainer.save_model(output_dir)
# print("Training complete. Model saved to", output_dir)