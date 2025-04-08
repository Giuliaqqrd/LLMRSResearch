from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed

save_directory = "/mnt/storage/huggingface/hub"

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir=save_directory)
set_seed(5)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device = 0, do_sample=True)
output = generator("", max_new_tokens=256)
print(output)
