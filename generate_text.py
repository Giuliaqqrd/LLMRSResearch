from huggingface_hub import InferenceClient
import sys

client = InferenceClient(
    base_url="http://localhost:40900/v1/",
)

# Ciclo per prendere input dall'utente fino a quando non inserisce 'exit'
while True:
    user_input = input("User: ")  # Chiedi all'utente di inserire un testo
    
    if user_input.lower() == "exit":  # Se l'utente digita "exit", interrompe il ciclo
        print("Exiting...")
        break

    # Invia la richiesta al modello
    output = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a recommender system."},
            {"role": "user", "content": user_input},  # Usa l'input dell'utente
        ],
        stream=True,
        max_tokens=1024,
        temperature = 0.3
    )

    for chunk in output:
        sys.stdout.write(chunk.choices[0].delta.content)
        sys.stdout.flush()
    
    print("\n\n")



