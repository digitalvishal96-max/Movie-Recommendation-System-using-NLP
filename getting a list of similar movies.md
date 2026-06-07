# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))

This line is a critical step in a **Content-Based Movie Recommendation System**.

```python
similarity_score = list(enumerate(similarity[index_of_the_movie]))
```

Let's break it down.

---

# Step 1: What is `similarity`?

Earlier, you probably created:

```python
similarity = cosine_similarity(feature_vectors)
```

Output:

```python
similarity.shape
```

```text
(4803, 4803)
```

Meaning:

* 4803 movies
* Each movie compared against every other movie

Example:

| Movie    | Avatar | Titanic | Iron Man |
| -------- | ------ | ------- | -------- |
| Avatar   | 1.00   | 0.15    | 0.62     |
| Titanic  | 0.15   | 1.00    | 0.10     |
| Iron Man | 0.62   | 0.10    | 1.00     |

---

# Step 2: What is `index_of_the_movie`?

Suppose the user enters:

```text
Iron Man
```

You find its index:

```python
index_of_the_movie = 68
```

---

# Step 3: What is

```python
similarity[index_of_the_movie]
```

If:

```python
index_of_the_movie = 68
```

Then:

```python
similarity[68]
```

returns the similarity scores of movie 68 against all movies.

Example:

```python
array([
    0.12,
    0.35,
    0.89,
    1.00,
    0.45
])
```

Interpretation:

| Movie Index | Similarity |
| ----------- | ---------- |
| 0           | 0.12       |
| 1           | 0.35       |
| 2           | 0.89       |
| 68          | 1.00       |
| 4           | 0.45       |

Notice:

```text
68 → 1.00
```

because a movie is 100% similar to itself.

---

# Step 4: What does `enumerate()` do?

```python
enumerate(similarity[68])
```

adds index numbers.

Example:

```python
[
 (0, 0.12),
 (1, 0.35),
 (2, 0.89),
 (68, 1.00),
 (4, 0.45)
]
```

Format:

```text
(movie_index, similarity_score)
```

---

# Step 5: Why convert to a list?

`enumerate()` creates an iterator.

To see and sort the results:

```python
similarity_score = list(
    enumerate(similarity[index_of_the_movie])
)
```

Now:

```python
print(similarity_score[:5])
```

Output:

```python
[
 (0, 0.12),
 (1, 0.35),
 (2, 0.89),
 (3, 0.67),
 (4, 0.45)
]
```

---

# Visualization

Suppose the user searched:

```text
Iron Man
```

Movie index:

```python
68
```

Then:

```python
similarity_score
```

becomes:

```python
[
 (68, 1.00),   # Iron Man
 (120, 0.95),  # Iron Man 2
 (310, 0.91),  # Avengers
 (450, 0.89),  # Captain America
 (221, 0.85),  # Thor
 ...
]
```

---

# What happens next?

Usually:

```python
sorted_similar_movies = sorted(
    similarity_score,
    key=lambda x: x[1],
    reverse=True
)
```

Sorts by similarity score.

Output:

```python
[
 (68, 1.00),
 (120, 0.95),
 (310, 0.91),
 (450, 0.89),
 ...
]
```

---

# Then get movie titles

```python
for movie in sorted_similar_movies[1:11]:
    
    index = movie[0]

    print(df.iloc[index]['title'])
```

Output:

```text
Iron Man 2
The Avengers
Captain America
Thor
Avengers: Age of Ultron
...
```

---

# Interview Explanation

If asked:

> Why use `enumerate(similarity[index_of_the_movie])`?

You can answer:

> The similarity matrix contains only similarity scores. By applying `enumerate()`, I pair each similarity score with its corresponding movie index. This allows me to sort movies by similarity and retrieve the associated movie titles from the dataset, forming the basis of the recommendation engine.

### Example

```python
similarity_score[:5]
```

might look like:

```python
[
 (0, 0.234),
 (1, 0.786),
 (2, 0.145),
 (3, 0.912),
 (4, 0.567)
]
```

where:

* First value = movie index
* Second value = similarity score

This structure is what enables the recommender to identify and rank the most similar movies.
