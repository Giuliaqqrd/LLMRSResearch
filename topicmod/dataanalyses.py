import pandas as pd
import nltk 
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def count_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    paragraphs = [p for p in text.split('\n\n') if p.strip()]  # Dividi per doppio newline e rimuovi vuoti
    
    print(f"Number of paragraphs: {len(paragraphs)}")
    return paragraphs

# Esempio di utilizzo
file_path = "/home/ubuntu/projects/LLMRSResearch/data/tune2/fulldataset.txt"  # Sostituisci con il percorso del tuo file
paragraphs = count_paragraphs(file_path)

data = []

for p in paragraphs:
    data.append(p)


df = pd.DataFrame(data)



print(df.head(10))

# for p in paragraphs:
#     print((len(nltk.word_tokenize(p))))

def count_tokens_by_section(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    sections = ["CORPORATE DOCUMENTS", "WIKIPEDIA", "EUROPEAN DOCUMENTS", "SCIENTIFIC PAPERS"]
    token_counts = {}
    
    current_section = None
    current_text = ""
    
    for line in text.split('\n'):
        line = line.strip()
        if line in sections:
            if current_section:
                token_counts[current_section] = len(word_tokenize(current_text))
            current_section = line
            current_text = ""
        else:
            current_text += " " + line
    
    if current_section:
        token_counts[current_section] = len(word_tokenize(current_text))
    
    return token_counts


token_counts = count_tokens_by_section(file_path)
print(token_counts)