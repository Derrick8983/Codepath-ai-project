# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**MoodMatch 1.0**

---

## 2. Intended Use  

Suggests songs based on a user's favorite genre and mood. Built for classroom exploration, not real-world use.

---

## 3. How the Model Works  

Each song is scored against user preferences. A genre match adds 2 points, a mood match adds 1, max score is 3. Songs are ranked and the top 5 are returned. Energy scoring was removed to keep the logic focused.

---

## 4. Data  

17 songs total. Started with 10 and added 7 to cover missing genres like hip-hop, classical, country, r&b, metal, reggae, and blues. Still a small catalog with no non-English music or older styles represented.

---

## 5. Strengths  

Works well when the user's genre and mood are in the catalog. My test profile (pop, happy) scored a perfect 3.00/3.00. Every result includes a plain reason so it is always clear why a song was picked.

---

## 6. Limitations and Bias  

Songs with no genre or mood match all score 0, making the lower results basically random. Genre matching is strict so "indie pop" and "pop" are treated as different. Tempo, danceability, and acousticness are unused.

---

## 7. Evaluation  

Tested with a pop/happy profile. Top results made sense. Removing energy caused everything outside the top matches to score 0, which showed how much one feature can shift the whole ranking.

---

## 8. Future Work  

Add energy back as a tiebreaker. Add a no repeat artist rule for more variety. Let users rate results so weights can adjust over time.

---

## 9. Personal Reflection  

I learned that recommender systems are really just scoring and sorting, the hard part is deciding what to measure. The most unexpected thing was how removing energy caused most songs to drop to 0, one small change completely flattened the results. It changed the way I think about apps like Spotify because I used to think they were reading my mind but now I know they are running the same basic idea just with way more features.
