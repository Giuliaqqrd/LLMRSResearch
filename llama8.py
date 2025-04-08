from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


save_directory = "/mnt/storage/huggingface/hub"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=save_directory)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=save_directory
)

while True:
    user_input = input("User: ")  # Chiedi all'utente di inserire un testo
    
    if user_input.lower() == "exit":  # Se l'utente digita "exit", interrompe il ciclo
        print("Exiting...")
        break

    messages = [
        {"role": "system", "content": "You are a recommender."},
        {"role": "user", "content": user_input},
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
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print()
    print("Answer: ",tokenizer.decode(response, skip_special_tokens=True))
    print()