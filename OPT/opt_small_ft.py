import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset, Features, Value

save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/optsmall"
base_model_id = "facebook/opt-125m"
data_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"

features = Features({ "text": Value("string") })
dataset = load_dataset("text", data_files={"train": data_file}, split="train")


def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True, )

model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", cache_dir=save_directory)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    cache_dir=save_directory
)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm = False)
# Set up the chat format with default 'chatml' format

print(tokenized_datasets)

optdataset = tokenized_datasets.train_test_split(test_size=0.2)

print(optdataset)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    warmup_ratio= 0.1,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    num_train_epochs= 10,
    logging_steps=10,
    logging_first_step=True,
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=optdataset["train"],
    eval_dataset=optdataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
)

model.config.use_cache = False

trainer.train()