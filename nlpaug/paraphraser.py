from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
import nltk
nltk.download('punkt') 

# Percorsi del file di input e di output
input_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/docexp.txt"
output_file = "/home/ubuntu/projects/LLMRSResearch/nlpaug/dataset_paraphrased.txt"

# Imposta il seed per la riproducibilità
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Carica il modello e il tokenizer
model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")
paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Imposta il dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
model = model.to(device)

# Funzione per generare la parafrasi
def generate_paraphrase(sentence):
    text = "paraphrase: " + sentence + " "
    
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1  # Numero di paraphrasi da generare
    )

    paraphrases = []
    for i, line in enumerate(beam_outputs):
        paraphrase = tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrases.append(paraphrase)
    return paraphrases

with open(input_file, "r", encoding="utf-8") as f:
    sentences = f.readlines()


with open(output_file, "w", encoding="utf-8") as output_f:
    for sentence in sentences:
        print("Originale: "+ sentence)
        sentence = sentence.strip()  # Rimuovi eventuali spazi vuoti e caratteri di nuova linea
        print("strip: "+ sentence)

        if sentence:
            print(f"Generando paraphrasi per: {sentence}")
            paraphrases = generate_paraphrase(sentence)
            for i, paraphrase in enumerate(paraphrases):
                output_f.write(f"{paraphrase}\n")
            output_f.write("\n")  # Aggiungi una linea vuota per separare le frasi
