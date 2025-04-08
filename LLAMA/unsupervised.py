from datasets import load_dataset, Features, Value

data_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"
features = Features({ "text": Value("string") })
dataset = load_dataset("text", data_files={"train": data_file})['train']

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



save_directory = "/mnt/storage/huggingface/hub"
output_dir = "/mnt/storage/lamapeft4"

base_model_id = "meta-llama/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", cache_dir=save_directory)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    cache_dir=save_directory
)
tokenizer.pad_token = tokenizer.eos_token

max_length = 512

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        prompt['text'],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

# print(tokenized_dataset)

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

import transformers
from datetime import datetime



trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=16,
        gradient_checkpointing=True,
        max_steps=1000,
        learning_rate=2e-5, 
        bf16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",       
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False 
trainer.train()