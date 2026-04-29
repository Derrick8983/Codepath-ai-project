import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

st.set_page_config(page_title="MoodMatch", page_icon="🎵", layout="centered")

st.title("🎵 MoodMatch Music Recommender")
st.caption("Describe the music you're in the mood for and Claude will find your best matches.")

# ── Sidebar: API key + index status ──────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    index_exists = (
        os.path.exists("data/song_embeddings.npy")
        and os.path.exists("data/song_index.json")
    )

    if index_exists:
        st.success("Song index ready")
    else:
        st.warning("Index not built yet")
        if st.button("Build Index"):
            from rag_indexer import build_index
            with st.spinner("Building index..."):
                build_index()
            st.rerun()

    st.divider()
    st.markdown("**Classic mode** scores songs by exact genre + mood match (max 3.0).")
    st.markdown("**RAG mode** uses semantic search + Claude to match any natural language description.")

# ── Main: tabs for both modes ─────────────────────────────────────────────────
tab_rag, tab_classic = st.tabs(["RAG Mode (Claude)", "Classic Mode"])

# RAG tab
with tab_rag:
    query = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g. chill music for late-night studying",
    )

    if st.button("Find Music", type="primary"):
        if not query.strip():
            st.warning("Enter a description first.")
        elif not index_exists:
            st.error("Build the song index first using the sidebar button.")
        else:
            try:
                from rag_retriever import retrieve
                from rag_recommender import generate_recommendations

                with st.spinner("Finding similar songs..."):
                    retrieved = retrieve(query, k=5)

                st.subheader("Retrieved Songs")
                for i, (song, score, desc) in enumerate(retrieved, 1):
                    with st.expander(
                        f"#{i}  {song['title']} by {song['artist']}  —  similarity: {score:.2f}"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Genre:** {song['genre']}")
                            st.write(f"**Mood:** {song['mood']}")
                            st.write(f"**Energy:** {float(song['energy']):.0%}")
                        with col2:
                            st.write(f"**Tempo:** {float(song['tempo_bpm']):.0f} BPM")
                            st.write(f"**Danceability:** {float(song['danceability']):.0%}")
                            st.write(f"**Acousticness:** {float(song['acousticness']):.0%}")

                st.subheader("Claude's Recommendation")
                with st.spinner("Generating with Claude..."):
                    response = generate_recommendations(query, retrieved)
                st.write(response)

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

# Classic tab
with tab_classic:
    st.markdown("Scores every song against a fixed pop/happy profile using genre (+2) and mood (+1) matching.")

    if st.button("Run Classic Recommender"):
        from recommender import load_songs, recommend_songs

        songs = load_songs("data/songs.csv")
        user_prefs = {
            "genre": "pop",
            "mood": "happy",
            "tempo_bpm": 120,
            "valence": 0.80,
            "danceability": 0.85,
            "likes_acoustic": False,
        }
        results = recommend_songs(user_prefs, songs, k=5)

        st.subheader("Top 5 Recommendations")
        for i, (song, score, explanation) in enumerate(results, 1):
            st.markdown(f"**#{i} — {song['title']}** by {song['artist']}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"Genre: {song['genre']}  |  Mood: {song['mood']}")
                st.progress(score / 3.0, text=f"Score: {score:.2f} / 3.00")
            with col2:
                st.info(f"Why: {explanation if explanation else 'no match'}")
            st.divider()
