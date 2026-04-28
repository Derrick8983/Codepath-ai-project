# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.
I have no idea how music reccomedation works in the real world but if i had to guess it woul dthat it uses the tempo and arttist to guess the mood and genre. And with records of previous sonds it mathces that to find the reccomended song. 
Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.
Algorithm Recipe

Scoring Rule — score_song(user_prefs, song)

Rule	Points	Logic
Genre match	+2.0	if song["genre"] == user_prefs["genre"]
Mood match	+1.0	if song["mood"] == user_prefs["mood"]
Energy similarity	+0.0 to 1.0	1.0 - abs(song["energy"] - user_prefs["energy"])
Returns: (score, reasons) — a number and a list of strings explaining why
Ranking Rule — recommend_songs(user_prefs, songs, k)

Loop through every song, call score_song on each
Collect (song, score, explanation) tuples
Sort by score descending
Return the top k
Supporting function — load_songs(csv_path)

Open the CSV with Python's csv.DictReader
Cast numeric columns (energy, tempo_bpm, valence, danceability, acousticness) from strings to float
Return a list of dicts
Max possible score per song: 4.0 (2.0 genre + 1.0 mood + 1.0 perfect energy match)
Ready to implement when you are.

Output: 
========================================
  TOP RECOMMENDATIONS FOR YOU
========================================

#1: Sunrise City by Neon Echo
    Genre: pop | Mood: happy
    Score: 3.98 / 4.00
    Why:   genre match, mood match, energy similarity 0.98

#2: Gym Hero by Max Pulse
    Genre: pop | Mood: intense
    Score: 2.87 / 4.00
    Why:   genre match, energy similarity 0.87

#3: Rooftop Lights by Indigo Parade
    Genre: indie pop | Mood: happy
    Score: 1.96 / 4.00
    Why:   mood match, energy similarity 0.96

#4: Night Drive Loop by Neon Echo
    Genre: synthwave | Mood: moody
    Score: 0.95 / 4.00
    Why:   energy similarity 0.95

#5: Gold Rush Flow by Street Cipher
    Genre: hip-hop | Mood: energetic
    Score: 0.92 / 4.00
    Why:   energy similarity 0.92

========================================


---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

