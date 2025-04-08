#Comandi per gestire accelerator
#CUDA_VISIBLE_DEVICES={gpus you gonna use} python -m torch.distributed.launch --nproc_per_node={the number of gpu used} \
#your_python_script.py

#export CUDA_VISIBLE_DEVICES=0,1

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import setup_chat_format
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

save_directory = "/mnt/storage/huggingface/hub"
data_file = "data/description_profile_instr.json"
output_dir = "/mnt/storage/tmp/fbnew"

dataset = load_dataset("json", data_files=data_file, split="train")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", cache_dir=save_directory)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", cache_dir=save_directory)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm = False)
# Set up the chat format with default 'chatml' format
model, tokenizer = setup_chat_format(model, tokenizer)

training_args = SFTConfig(
    max_seq_length=512,
    output_dir= output_dir,
)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset,
    args=training_args,
    data_collator = data_collator,
)
trainer.train()

trainer.save_model(output_dir)