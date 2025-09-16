import numpy as np, faiss, pickle
from recommender_backend import embed, recommend

index = faiss.read_index("netflix.index")
with open("netflix.pkl","rb") as f:
    movies = pickle.load(f)

# recommend() now only embeds the *query*
    def recommend(query, k=5):
        q = embed([query])        # only one embedding call
        sims, idxs = index.search(q, k)
        return [(movies.iloc[int(i)]["title"], float(sims[0][j])) for j,i in enumerate(idxs[0])]

# demo
    for title,score in recommend("dream-heist sci-fi with layered realities", k=3):
        print(f"{score:.3f}  {title}")