"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Taste profile: upbeat pop listener who enjoys dancing and positivity
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "tempo_bpm": 120,      # comfortable around 120 BPM (dance-friendly)
        "valence": 0.80,       # wants positive, feel-good songs
        "danceability": 0.85,  # high danceability is a priority
        "likes_acoustic": False # prefers produced/electronic sound over acoustic
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
