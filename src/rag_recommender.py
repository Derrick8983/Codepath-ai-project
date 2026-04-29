import anthropic
from typing import List, Tuple

SYSTEM_PROMPT = """You are a music recommendation assistant with deep knowledge of how audio features relate to listening experiences. When given a user's natural language music request and a list of candidate songs retrieved by semantic search, analyze how well each song matches what the user is looking for.

Consider mood, energy needs, genre preferences, and contextual cues (working out, studying, relaxing, driving, etc.). Rank the songs from best to worst match and explain your reasoning in plain language. Be specific — reference each song's actual features (tempo, mood, genre, energy) rather than speaking in generalities."""


def generate_recommendations(query: str, retrieved: List[Tuple[dict, float, str]]) -> str:
    client = anthropic.Anthropic()

    songs_text = "\n".join(
        f"{i + 1}. \"{s['title']}\" by {s['artist']} "
        f"[{s['genre']}, {s['mood']}, {float(s['energy']):.0%} energy, {float(s['tempo_bpm']):.0f} BPM]\n"
        f"   {desc}"
        for i, (s, score, desc) in enumerate(retrieved)
    )

    user_message = (
        f'User request: "{query}"\n\n'
        f"Retrieved songs:\n{songs_text}\n\n"
        f"Rank these songs for this user and explain why each fits or doesn't fit their request."
    )

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()
        return next(b.text for b in response.content if b.type == "text")
