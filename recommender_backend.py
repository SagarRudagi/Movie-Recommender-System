import pandas as pd
import numpy as np
import faiss, pickle
from ollama import Client

#Filtering the data to only the relevant columns

amazonData = pd.read_csv("amazon_prime_titles.csv")
amazonData = amazonData[["title","listed_in","cast","description"]]
amazonData.columns = ["title", "genre", "cast", "plot"]

netflixData = pd.read_csv("netflix_titles.csv")
netflixData = netflixData[["title","listed_in","cast","description"]]
netflixData.columns = ["title", "genre", "cast", "plot"]
netflixData['cast'] = netflixData['cast'].fillna('Unknown')

huluData = pd.read_csv("hulu_titles.csv")
huluData = huluData[["title","listed_in","cast","description"]]
huluData.columns = ["title", "genre", "cast", "plot"]
huluData['cast'] = huluData['cast'].fillna('Unknown')

disneyData = pd.read_csv("disney_plus_titles.csv")
disneyData = disneyData[["title","listed_in","cast","description"]]
disneyData.columns = ["title", "genre", "cast", "plot"]

#Combining all the data into one dataframe
allData = pd.concat([amazonData, netflixData, huluData, disneyData], ignore_index=True)

#Filling in missing values
allData['cast'] = allData['cast'].fillna('Unknown')

#Function to create a text representation of a movie
def movie_text(df):
    corpus = []
    for _, m in df.iterrows():
        corpus.append(f"{m['title']} | {m['genre']} | Cast: {m['cast']} | {m['plot']}")
    return corpus

amazonCorpus = movie_text(amazonData)
netflixCorpus = movie_text(netflixData)
huluCorpus = movie_text(huluData)
disneyCorpus= movie_text(disneyData)
allCorpus = movie_text(allData)

EMBED_MODEL = "nomic-embed-text"      # or "mxbai-embed-large"
client = Client(host="http://localhost:11434")

def embed(texts):
    """
    Ollama's Python client embeds one prompt at a time.
    We loop, collect vectors, and L2-normalize for cosine.
    """
    vecs = []
    for t in texts:
        resp = client.embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(resp["embedding"])
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # cosine when combined with inner product
    return arr

E = embed(netflixCorpus)
index = faiss.IndexFlatIP(E.shape[1])  # inner product on normalized vectors = cosine similarity
index.add(E)

def recommend(query, k=5):
    q = embed([query])
    sims, idxs = index.search(q, k)
    return [(netflixData.iloc[int(i)]["title"], float(sims[0][j])) for j,i in enumerate(idxs[0])]

#Compute and store the model after final compute
faiss.write_index(index, "netflix.index")       # save the FAISS index
np.save("movie_vectors.npy", E)                # optional: save raw embeddings
with open("netflix.pkl","wb") as f:
    pickle.dump(netflixData, f)