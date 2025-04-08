import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

nltk.download('punkt')

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    sentences = sent_tokenize(text)  # Divide il testo in frasi
    return sentences

def cluster_text(sentences, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    return X, clusters

def plot_clusters(X, clusters, save_path="/home/ubuntu/projects/LLMRSResearch/data/plot/cluster_plot.png"):
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 6))
    colors = np.array(["red", "blue", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"])
    
    for i in range(len(reduced_X)):
        plt.scatter(reduced_X[i, 0], reduced_X[i, 1], color=colors[clusters[i]])
    
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Text Clustering")
    plt.savefig(save_path)
    plt.show()

# Esempio di utilizzo
file_path = "/home/ubuntu/projects/LLMRSResearch/data/tune2/docexp.txt"  # Sostituisci con il percorso del tuo file
txt = load_text(file_path)
sentences = preprocess_text(txt)
X, clusters = cluster_text(sentences)
plot_clusters(X, clusters)
