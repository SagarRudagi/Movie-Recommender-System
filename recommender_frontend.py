
import streamlit as st
import os
import random
import pickle
import faiss

from recommender_backend import ensure_index, embed

# Simple, minimal Streamlit frontend
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommender System")

st.write("") 
st.write("") 

# Display all images from the Posters/ folder side-by-side immediately after the title
posters_dir = 'Posters'
if os.path.isdir(posters_dir):
    imgs = []
    for ext in ('.jpg', '.jpeg', '.png', '.webp'):
        imgs.extend(sorted([os.path.join(posters_dir, f) for f in os.listdir(posters_dir) if f.lower().endswith(ext)]))
    if imgs:
        # limit columns to a reasonable count if there are many images
        cols = st.columns(len(imgs))
        for c, img_path in zip(cols, imgs):
            try:
                c.image(img_path, use_column_width=True)
            except Exception:
                c.write(os.path.basename(img_path))
    else:
        st.info("No poster images found in the Posters/ folder.")
else:
    st.info("Posters/ folder not found.")


st.write("")   # leaves one blank line
st.write("") 
st.write("")   
st.write("") 

st.subheader("Powered by Ollama, FAISS, and Streamlit")


st.markdown(
    """
    <p style='font-size:20px;'>
    The backend contains the data-preparation and indexing engine for a multi-platform movie recommender. It loads the raw CSV listings from Amazon Prime, Netflix, Hulu, and Disney+, keeps only the key fields (title, genre, cast, plot) and fills any missing cast values with “Unknown”. For each service it creates a single descriptive string per title—combining title, genres, main cast and plot—so every show or film has one clean piece of text ready for embedding.
    </p>
    <p style='font-size:20px;'>
    Those descriptions are converted into vector embeddings using a local Ollama model (by default nomic-embed-text) and L2-normalized so that FAISS can perform fast cosine-similarity search. For every dataset (and for the combined “all” catalogue) the script builds and saves a FAISS index along with the raw embeddings and a pickled DataFrame of the original metadata. Later, a front-end or API can simply load these artifacts, embed a user query once, and retrieve the most similar titles in milliseconds without re-processing the data.
    </p>
    """,
    unsafe_allow_html=True
)


def load_movies(platform_key):
    p = f"{platform_key}.pkl"
    if os.path.exists(p):
        with open(p, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception:
                return None
    return None


def extract_title(item):
    if item is None:
        return ''
    try:
        # pandas row/series
        if hasattr(item, 'get') and item.get('title') is not None:
            return str(item.get('title'))
    except Exception:
        pass
    s = str(item)
    return s.split('|', 1)[0].strip() if '|' in s else s


def poster_for_title(title):
    if not title:
        return None
    safe = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    for ext in ('.jpg', '.jpeg', '.png'):
        path = os.path.join('posters', safe + ext)
        if os.path.exists(path):
            return path
    return None


st.write("")   # leaves one blank line
st.write("") 
st.write("")   
st.write("")

st.subheader('Get recommendations')
display_options = ['Netflix', 'Amazon', 'Hulu', 'Disney', 'All']
display = st.selectbox('Platform', options=display_options)
platform_key = display.lower()

movies_obj = load_movies(platform_key)
titles = []
if movies_obj is None:
    st.warning(f"No movie list found for '{platform_key}'. Create {platform_key}.pkl to enable selection.")
else:
    if hasattr(movies_obj, 'columns') and 'title' in movies_obj.columns:
        titles = movies_obj['title'].astype(str).tolist()
    elif isinstance(movies_obj, list):
        titles = [extract_title(x) for x in movies_obj]

titles = [extract_title(t) for t in titles]
choice = st.selectbox('Pick a movie you like', options=titles[:] if titles else [])

k = st.slider('Number of recommendations', min_value=1, max_value=10, value=3)
k += 1 #Add 1 to k to account for the input movie itself


if st.button('Recommend'):
    if not choice:
        st.error('Please pick a movie')
    else:
        with st.spinner('Finding similar titles...'):
            ensure_index(platform_key)
            idx = faiss.read_index(f"{platform_key}.index")

            # try to find the full metadata row for the chosen title
            full_text = choice
            if movies_obj is not None:
                # if dataframe
                try:
                    if hasattr(movies_obj, 'columns') and 'title' in movies_obj.columns:
                        rows = movies_obj[movies_obj['title'].astype(str) == choice]
                        if len(rows) > 0:
                            r = rows.iloc[0]
                            full_text = f"{r['title']} | {r.get('genre','')} | Cast: {r.get('cast','')} | {r.get('plot','')}"
                    else:
                        # list-like
                        for it in movies_obj:
                            if extract_title(it) == choice:
                                full_text = str(it)
                                break
                except Exception:
                    pass

                q = embed([full_text])
                # use user-specified number of recommendations
                
                D, I = idx.search(q, k)
                
                st.success('Recommendations')
                for j in range(1, min(k, len(I[0]))):
                    i = int(I[0][j])
                    score = float(D[0][j])
                    try:
                        if hasattr(movies_obj, 'iloc'):
                            row = movies_obj.iloc[i]
                            st.write(row)
                            title = row['title']
                            genre = row.get('genre', '')
                            cast = row.get('cast', '')
                            plot = row.get('plot', '')
                        else:
                                item = movies_obj[i]
                                # list-style items are stored as pipe-separated strings like:
                                # "Title | Genre | Cast: ... | Plot"
                                s = str(item)
                                parts = [p.strip() for p in s.split('|')]
                                title = parts[0] if len(parts) > 0 else extract_title(item)
                                genre = parts[1] if len(parts) > 1 else ''
                                # attempt to find a part starting with 'Cast:'
                                cast = ''
                                plot = ''
                                for p in parts[1:]:
                                    if p.lower().startswith('cast:'):
                                        cast = p.split(':', 1)[1].strip()
                                    elif not cast and 'cast' in p.lower() and ':' in p:
                                        # fallback
                                        cast = p.split(':', 1)[1].strip()
                                # last part as plot if available
                                if len(parts) > 2:
                                    plot = parts[-1]
                    except Exception:
                        title = 'Unknown'
                        genre = cast = plot = ''

                    st.markdown(f"### {title}")
                    if genre:
                        st.write(f"**Genre:** {genre}")
                    if cast:
                        st.write(f"**Cast:** {cast}")
                    if plot:
                        st.write(f"**Plot:** {plot}")
                    p = poster_for_title(title)
                    if p:
                        st.image(p, width=180)
                    st.write(f"**Score:** {score:.3f}")
                    st.markdown('---')
