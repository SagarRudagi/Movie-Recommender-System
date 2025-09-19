# 🎬 Movie Recommender System

A lightweight, local-first movie recommender demo using local embeddings (Ollama), FAISS vector search, and a Streamlit UI. Built for fast, interactive recommendations across multiple streaming catalogs (Netflix, Amazon Prime, Hulu, Disney+ and a combined "all" dataset).

---

## 🚀 Features

- Preprocesses multiple catalog CSVs into clean textual descriptions (Title | Genre | Cast | Plot).
- Computes and persists vector embeddings and FAISS indexes for fast similarity search.
- Streamlit frontend with platform selector, movie picker, posters gallery, and configurable number of recommendations.
- Local-first design using Ollama for embeddings (can be tunneled for remote demos using ngrok/cloudflared).
- Helper scripts and documentation to run locally and share a public demo.

---

## 📁 Repo layout

- `app.py` — Streamlit frontend UI
- `backend.py` — data prep, embedding wrapper, index builder (`ensure_index`, `build_index`, `build_all_indexes`)
- `*.csv` — raw title data per platform (Netflix, Amazon, Hulu, Disney)
- `*.pkl`, `*_vectors.npy`, `*.index` — generated artifacts (pickles, embeddings, FAISS indexes)
- `Posters/` — poster images used in the UI
- `start_all.sh` — helper to start Streamlit + optional ngrok tunnels
- `requirements.txt` — Python dependencies
- `documentation.txt` — extended docs and ngrok hosting guide

---

## 🧠 How it works (high level)

1. Backend reads CSVs and creates a single descriptive string per title: `"Title | Genre | Cast: ... | Plot"`.
2. These texts are embedded (using a model served by Ollama) and the embeddings are saved to `{platform}_vectors.npy`.
3. FAISS index is built from the saved vectors and written to `{platform}.index` for fast nearest-neighbor search.
4. At query time the frontend:
   - Composes a `full_text` for the selected movie,
   - Calls the embedding API once to get a query vector,
   - Runs `faiss.index.search(query_vector, k)` to get top-k neighbors,
   - Maps neighbor indices to metadata in `{platform}.pkl` and displays results.

This avoids an O(N) per-query embedding step and reduces runtime from minutes to sub-second/second latency.

---

## ⚙️ Dependencies

Install dependencies in a virtualenv.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:
- streamlit
- numpy
- pandas
- faiss-cpu
- ollama
- requests

Note: `ngrok` is not a pip package — install via Homebrew (`brew install ngrok/ngrok/ngrok`) or download from https://ngrok.com/.

---

## 🧩 Quick start (local)

1. Ensure Ollama is running locally and the embedding model (e.g. `nomic-embed-text`) is available.
2. Build indices (optional, the app will build lazily if missing):

```bash
python3 recommender_backend.py
```

3. Run the app:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Or use the helper (starts Streamlit and optionally ngrok tunnels):

```bash
./start_all.sh
```

---

## 🌐 Share a demo (ngrok)

To share the local demo publicly (quick & free), expose Streamlit via ngrok:

```bash
ngrok http 8501
```

If the app needs to call your local Ollama server from the public internet, also run:

```bash
ngrok http 11434
export OLLAMA_URL="https://<your-ngrok-ollama-url>"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Security notes: anyone with the ngrok URL can access the app — consider adding a simple password gate or using Cloudflare Access.

---

## ✅ Tips & next steps

- Normalize pickles to DataFrames with columns `title, genre, cast, plot` for easier downstream work.
- Precompute embeddings on a machine once and reuse the generated files to avoid repeated heavy computation.
- If you need a persistent free hosting option, refactor to use a hosted embeddings API or precompute vectors and deploy only the frontend and FAISS artifacts.


***

