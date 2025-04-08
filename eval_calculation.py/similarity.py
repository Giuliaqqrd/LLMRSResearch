from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Modello per embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Risposte ufficiali e generate
official_answers = [" A sustainability champion in the energy sector is a dedicated and knowledgeable advocate, driven by a commitment to combating climate change and environmental degradation. Their passion fuels their mission to promote renewable energy and advocate for energy efficiency, contributing to the transition toward a sustainable future. With an in-depth understanding of the energy ecosystem and its key players, they excel at identifying and driving impactful actions that help reduce carbon emissions and pollution. Equipped with resources such as solar panel installations and advanced smart home technology, they demonstrate sustainable energy solutions in action, inspiring others to adopt greener practices. Their unique combination of values, expertise, and resources makes them a powerful force in the fight against climate change—an influential voice that motivates a shift toward a cleaner, more sustainable energy landscape."]
generated_answers = ["Sustainability Champion is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability. It is a program that helps organizations to improve their sustainability."]

# Ottenere embeddings
official_emb = model.encode(official_answers)
generated_emb = model.encode(generated_answers)

# Calcolare la similarità coseno
similarity = cosine_similarity(official_emb, generated_emb)

print("Cosine Similarity:", similarity[0][0])
