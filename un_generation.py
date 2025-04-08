import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

save_directory = "/mnt/storage/huggingface/hub"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model_id = "meta-llama/Meta-Llama-3-8B"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=save_directory
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, cache_dir=save_directory)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "/mnt/storage/lamapeft4/checkpoint-1000")

eval_prompt = """ 
Briefly describe the benefits a sustainability champion has when entering an energy community."""
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=512, repetition_penalty=1.15)[0], skip_special_tokens=True))