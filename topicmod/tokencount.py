import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def count_tokens_in_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Tokenizza il testo intero
    tokens = word_tokenize(text)
    
    # Restituisce il numero totale di token
    return len(tokens)

# Esempio di utilizzo
file_path = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"  # Sostituisci con il percorso del tuo file
token_count = count_tokens_in_document(file_path)

print(f"Total token count in the document: {token_count}")
