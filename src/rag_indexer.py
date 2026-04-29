import csv
import json
import numpy as np
from pathlib import Path


def song_to_text(song: dict) -> str:
    energy = float(song["energy"])
    tempo = float(song["tempo_bpm"])
    valence = float(song["valence"])
    danceability = float(song["danceability"])
    acousticness = float(song["acousticness"])

    energy_desc = "high-energy" if energy > 0.7 else ("moderate-energy" if energy > 0.4 else "low-energy")
    tempo_desc = "fast" if tempo > 130 else ("moderate-tempo" if tempo > 90 else "slow")
    valence_desc = "uplifting and positive" if valence > 0.7 else ("mixed" if valence > 0.4 else "melancholic or somber")
    dance_desc = "highly danceable" if danceability > 0.7 else ("moderately danceable" if danceability > 0.4 else "not very danceable")
    sound_desc = "acoustic" if acousticness > 0.6 else "produced or electronic"

    return (
        f"'{song['title']}' by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"{energy_desc.capitalize()}, {tempo_desc} tempo at {tempo:.0f} BPM, "
        f"{valence_desc} feel, {dance_desc}, {sound_desc} sound."
    )


def build_index(csv_path: str = "data/songs.csv", output_dir: str = "data") -> None:
    from sentence_transformers import SentenceTransformer

    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(dict(row))

    descriptions = [song_to_text(s) for s in songs]

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True)

    output_path = Path(output_dir)
    np.save(output_path / "song_embeddings.npy", embeddings)
    with open(output_path / "song_index.json", "w") as f:
        json.dump({"songs": songs, "descriptions": descriptions}, f, indent=2)

    print(f"Indexed {len(songs)} songs → {output_dir}/song_embeddings.npy + song_index.json")


if __name__ == "__main__":
    build_index()
