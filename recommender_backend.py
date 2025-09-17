import pandas as pd
import numpy as np
import faiss, pickle, os
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

dataDict = {
    'amazon': (amazonCorpus, amazonData),
    'netflix': (netflixCorpus, netflixData),
    'hulu': (huluCorpus, huluData),
    'disney': (disneyCorpus, disneyData),
    'all': (allCorpus, allData),
}


def build_index(name, corpus, df, embed_fn=embed):
    """Compute embeddings and write index + artifacts for a single dataset."""
    vec_path = f"{name}_vectors.npy"
    idx_path = f"{name}.index"
    pkl_path = f"{name}.pkl"

    E = embed_fn(corpus)
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    faiss.write_index(index, idx_path)
    np.save(vec_path, E)
    with open(pkl_path, 'wb') as f:
        pickle.dump(df, f)


def ensure_index(name):
    """Ensure the named index and pickle exist. Build lazily if missing."""
    if name not in dataDict:
        raise KeyError(f"Unknown dataset: {name}")
    corpus, df = dataDict[name]
    vec_path = f"{name}_vectors.npy"
    idx_path = f"{name}.index"
    pkl_path = f"{name}.pkl"

    # create pickle if missing
    if not os.path.exists(pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(df, f)

    # build index and vectors if missing
    if not (os.path.exists(vec_path) and os.path.exists(idx_path)):
        print(f"Building index for {name} (this may take a while)...")
        build_index(name, corpus, df)


def build_all_indexes():
    """Convenience: build all indexes. Use sparingly."""
    for name, (corpus, df) in dataDict.items():
        ensure_index(name)