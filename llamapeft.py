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

save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/lamapeft2"

PAD_TOKEN = "<|pad|>"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NEW_MODEL = "Llama-3-8B-RS"

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
data_file = "data/tune2/unique_train_data.json"
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
df = df.sample(6000)
#print(df.shape)

# train, temp = train_test_split(df, test_size=0.2)
# val, test = train_test_split(temp, test_size=0.2)

# #print(len(train)/len(df), len(val)/len(df), len(test)/len(df))

# train.sample(n=4000).to_json("data/train2.json", orient="records", lines=True)
# val.sample(n=500).to_json("data/val2.json", orient="records", lines=True)
# test.sample(n=100).to_json("data/test2.json", orient="records", lines=True) 

train_set = "data/test2.jsonl"
validation_set = "data/val2.jsonl"
test_set = "data/test2.jsonl"

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

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
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

sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    max_seq_length=512,
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,
    save_strategy="steps",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
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
trainer.save_model(output_dir)