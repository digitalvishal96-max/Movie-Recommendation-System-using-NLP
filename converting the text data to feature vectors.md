converting the text data to feature vectors

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

print(feature_vectors)

explain -

<Compressed Sparse Row sparse matrix of dtype 'float64'
	with 124266 stored elements and shape (4803, 17318)> 

# GPT RESPONSE

This output is one of the most important things to understand in a **Content-Based Recommendation System**.

You ran:

```python
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)
```

and got:

```text
<Compressed Sparse Row sparse matrix of dtype 'float64'
with 124266 stored elements and shape (4803, 17318)>
```

Let's decode it.

---

# 1. What is `shape (4803, 17318)`?

Think of it as:

```text
Rows    = Movies
Columns = Unique Words
```

So:

```text
4803 rows     = 4803 movies
17318 columns = 17,318 unique words/tokens
```

Example:

| Movie         | action | adventure | batman | romance | war  |
| ------------- | ------ | --------- | ------ | ------- | ---- |
| Avatar        | 0.12   | 0.35      | 0.00   | 0.00    | 0.08 |
| Titanic       | 0.00   | 0.00      | 0.00   | 0.62    | 0.00 |
| Batman Begins | 0.20   | 0.10      | 0.50   | 0.00    | 0.00 |

Each movie becomes a vector of length **17,318**.

---

# 2. What does `124266 stored elements` mean?

Not every movie contains every word.

For example:

```text
Movie 1:
action adventure fantasy

Movie 2:
romance drama

Movie 3:
batman hero gotham
```

Most cells are:

```text
0
```

Only a few contain TF-IDF values.

Instead of storing:

```text
4803 × 17318
=
83,181,354 values
```

Scikit-learn stores only the non-zero values.

That's why:

```text
Stored elements = 124,266
```

instead of 83 million values.

Huge memory savings.

---

# 3. What is a Sparse Matrix?

Normal matrix:

```text
[0.12 0.00 0.55 0.00]
[0.00 0.00 0.00 0.78]
[0.00 0.22 0.00 0.00]
```

Most values are zero.

A sparse matrix stores only:

```text
Position → Value

(0,0) → 0.12
(0,2) → 0.55
(1,3) → 0.78
(2,1) → 0.22
```

Much more efficient.

---

# 4. What do these coordinates mean?

Example from your output:

```text
(0, 201)    0.0786
```

Means:

```text
Movie Row = 0
Word Column = 201
TF-IDF Score = 0.0786
```

---

Another:

```text
(0, 3678)   0.2139
```

Means:

```text
Movie #0 contains Word #3678
with importance score 0.2139
```

---

# 5. What is `dtype='float64'`?

The TF-IDF scores are decimal values:

```text
0.0786
0.2139
0.5551
```

stored as:

```python
float64
```

which is standard numerical precision.

---

# 6. How can I see the actual words?

### Vocabulary size

```python
len(vectorizer.get_feature_names_out())
```

Output:

```text
17318
```

---

### First 20 words

```python
vectorizer.get_feature_names_out()[:20]
```

Example:

```text
['007', '10', '100', '13th', '18th', ...]
```

---

### Word at column 201

```python
vectorizer.get_feature_names_out()[201]
```

Now you'll know what:

```text
(0, 201) = 0.0786
```

actually represents.

---

# 7. View Movie #0 as a DataFrame

```python
feature_df = pd.DataFrame.sparse.from_spmatrix(
    feature_vectors,
    columns=vectorizer.get_feature_names_out()
)

feature_df.head(1)
```

This converts the sparse matrix into a readable table.

---

# 8. Why are we doing all this?

Machine learning cannot understand:

```text
Action Adventure Fantasy Science Fiction
```

It understands numbers.

TF-IDF converts text into numerical vectors:

```text
Movie 1 → [0.12, 0.44, 0.00, ...]
Movie 2 → [0.00, 0.00, 0.65, ...]
Movie 3 → [0.20, 0.10, 0.00, ...]
```

Then:

```python
cosine_similarity(feature_vectors)
```

compares these vectors to find similar movies.

---

## ML Engineer Interpretation

Your output tells us:

```text
Movies in dataset      : 4,803
Unique text features   : 17,318
Non-zero TF-IDF values : 124,266
Representation         : Sparse Matrix
```

This is exactly what we expect before computing:

```python
similarity = cosine_similarity(feature_vectors)
```

which becomes the foundation of the recommendation engine.
