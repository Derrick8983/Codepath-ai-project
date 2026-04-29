"""
Command line runner for the Music Recommender.

Classic mode (no args):
    python src/main.py

RAG mode (natural language query):
    python src/main.py --query "something chill for late-night studying"
"""
import sys
from typing import Optional

from recommender import load_songs, recommend_songs


def main() -> None:
    query = _parse_query()
    if query:
        _run_rag_mode(query)
    else:
        _run_classic_mode()


def _parse_query() -> Optional[str]:
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def _run_rag_mode(query: str) -> None:
    from rag_retriever import retrieve
    from rag_recommender import generate_recommendations

    print("\n" + "=" * 40)
    print("  RAG MUSIC RECOMMENDATIONS")
    print(f'  "{query}"')
    print("=" * 40 + "\n")

    print("Retrieving similar songs...")
    retrieved = retrieve(query, k=5)

    print("Generating recommendations with Claude...\n")
    response = generate_recommendations(query, retrieved)
    print(response)
    print("\n" + "=" * 40)


def _run_classic_mode() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "tempo_bpm": 120,
        "valence": 0.80,
        "danceability": 0.85,
        "likes_acoustic": False,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 40)
    print("  TOP RECOMMENDATIONS FOR YOU")
    print("=" * 40)
    for i, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{i}: {song['title']} by {song['artist']}")
        print(f"    Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"    Score: {score:.2f} / 3.00")
        print(f"    Why:   {explanation}")
    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
