import nlpaug.augmenter.word as naw
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt') 

input_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/docexp.txt"   
output_file = "/home/ubuntu/projects/LLMRSResearch/nlpaug/dataset_augmented.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

sentences = nltk.sent_tokenize(text)

aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=3, aug_max=10, aug_p=0.1, lang='eng', 
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False, 
                     verbose=0)

augmented_sentences = [aug.augment(sentence)[0] for sentence in sentences]

 

with open(output_file, "w", encoding="utf-8") as f:
    for sentence in augmented_sentences:
        f.write(sentence + "\n")

print(f"Dataset aumentato salvato in {output_file}")