from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from trl import SFTTrainer, SFTConfig, setup_chat_format

data_file = "data/description_profile_instr.json"
save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/tmp"

#dataset = load_dataset("json", data_files = data_file, split= "train")
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(guanaco_dataset, split="train")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=save_directory)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=save_directory)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.resize_token_embeddings(len(tokenizer))
print("Aggiornamento tokenizer: ",len(tokenizer))
#print(dataset['train'][0])
model, tokenizer = setup_chat_format(model, tokenizer)

# def tokenize_function(example):
#     return tokenizer(
#         example['prompt'],
#         text_pair=example['completion'],
#         max_length=512,
#         truncation=True,
#         padding="max_length"
#     )

# Applica la tokenizzazione al dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Controlla un esempio tokenizzato
# print(tokenized_dataset[0])
# print(tokenizer.special_tokens_map)
# print(tokenizer.convert_ids_to_tokens([32001]))

training_args = SFTConfig(
    max_seq_length=512,
    output_dir= output_dir,
    per_device_train_batch_size=4
)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()


