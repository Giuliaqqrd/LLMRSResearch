import nltk

input_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/docexp.txt"   
output_file = "/home/ubuntu/projects/LLMRSResearch/nlpaug/dataset_sent.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()




sentences = nltk.sent_tokenize(text)
print(sentences[0])

# with open(output_file, "w", encoding="utf-8") as f:
#     for sentence in sentences:
#         f.write(sentence + "\n")

# print(f"Dataset aumentato salvato in {output_file}")