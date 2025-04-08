from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.optim as optim

device_count = torch.cuda.device_count()
print(f'Numero di GPU disponibili: {device_count}')


# Percorsi e configurazioni
save_directory = "/mnt/storage/huggingface/hub"
data_file = "data/description_profile_instr.json"
output_dir = "/mnt/storage/tmp"
max_length = 512
batch_size = 10

# Caricamento dataset
dataset = load_dataset("json", data_files=data_file, split="train")
print("Esempio di dataset iniziale:", dataset[0])

# Configurazione del tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=save_directory)
tokenizer.pad_token = tokenizer.eos_token  # Imposta il padding come EOS token

# Template Alpaca per i prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Prompt:
{}

### Completion:
{}"""

# Funzione per formattare i prompt
def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    outputs = examples["completion"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Formattazione del dataset
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
print("Esempio di dataset formattato:", formatted_dataset[0])

# Funzione di tokenizzazione
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation= True,
        return_tensors="pt",
    )
    return tokenized

# Tokenizzazione del dataset
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
print("Esempio di dataset tokenizzato:", tokenized_dataset[0])

# Filtra campioni vuoti (se presenti)
def filter_empty(examples):
    return len(examples["text"]) > 0

filtered_dataset = tokenized_dataset.filter(filter_empty)
print("Dataset filtrato, lunghezza:", len(filtered_dataset))

# Caricamento del modello
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=save_directory)
model = nn.DataParallel(model)
model = model.cuda()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Configurazione del Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=filtered_dataset,
    data_collator = data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        output_dir=output_dir,
        remove_unused_columns=False
    ),
)

# Avvio dell'addestramento
trainer.train()
