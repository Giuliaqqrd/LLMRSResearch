import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

PAD_TOKEN = "<|pad|>"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "/mnt/storage/lama8instr"
save_directory = "/mnt/storage/huggingface/hub"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, cache_dir=save_directory)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
#tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    cache_dir=save_directory
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
print("Adding pad: ",tokenizer.pad_token, tokenizer.pad_token_id)
tokenizer.convert_tokens_to_ids(PAD_TOKEN)
# Adapter LoRA
model = PeftModel.from_pretrained(model, LORA_PATH)

model.eval()



while True:
    user_input = input("User: ")  # Chiedi all'utente di inserire un testo
    
    if user_input.lower() == "exit":  # Se l'utente digita "exit", interrompe il ciclo
        print("Exiting...")
        break



    messages = [
        {
            "role": "system", 
            "content": "You are a recommender. Generate a long and engagement user profile description focusing on level of expertise and values."
            },
        {
            "role": "user", 
            "content": user_input},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.7,
    )


    response = outputs[0][input_ids.shape[-1]:]
    print()
    answer = tokenizer.decode(response, skip_special_tokens=True)
   
    print("Answer: ", answer)
    print()

 
