# DPVS: Deterministic Positional Vectorization of Strings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast approximate string matching library** that turns words into compact vectors so you can find the closest match, even when the query is riddled with typos! It’s like fuzzy matching on steroids: sub‑millisecond lookup times and linear memory scaling.

## Why DPVS?

I built DPVS because I needed to search through a large vocabulary with queries that often contained swapped letters, missing characters, and random insertions. Traditional edit‑distance algorithms were accurate but far too slow for real‑time use. Faster alternatives like SymSpell were quick, but they fell apart on transpositions ("lorem" → "olrem") and deletions ("ipsum" → "isum"). Their memory footprint also exploded with higher error tolerances!

DPVS solves both problems:

- **Speed:** FAISS‑powered HNSW index gives you ~14 000 queries per second **on a laptop CPU** (and even faster with a GPU!).
- **Accuracy:** It correctly catches 98.6 % of transpositions errors and 67.3 % of deletions, often beating SymSpell by a wide margin.
- **Memory:** The index size grows **linearly** with dictionary size. No exponential blow‑up from edit distance.

If you’ve ever wished for a fuzzy matching distance algorithm that runs at hash‑table speed, DPVS is for you!

---

## Features

- **Deterministic vectorisation:** Same word always gives the same vector.
- **All‑in‑one fuzzy search:** Handles substitutions, insertions, deletions, and swaps without any extra steps.
- **FAISS‑backed:** Choose between CPU or GPU indexing for massive throughput.
- **Linear memory:** Index size is `O(dictionary_size * vector_size)`, regardless of error tolerance.
- **Simple API:** Build, save, load, and query with a few lines of Python.
- **Typo‑tolerant, not a typo‑corrector:** Use it anywhere you need approximate string matching: record linkage, OCR post‑processing, fuzzy search, or duplicate detection.

---

## How It Works

DPVS converts each word into a fixed‑length vector that encodes four complementary views of the string:

1. **Character frequency:** How often each letter appears.
2. **Average position:** Where each letter tends to sit in the word.
3. **Preceding characters:** Weighted influence of letters that come before.
4. **Succeeding characters:** Weighted influence of letters that come after.

All vectors are normalized by word length, so "apple" and "apples" end up close together.

After vectorizing the dictionary, we build a **FAISS HNSW index** using Manhattan (L1) distance. A query vector is searched in this index, and the nearest neighbours become your top‑k candidates. The whole process is deterministic, interpretable, and very, very fast.

---

## Benchmark Highlights

Here’s a quick comparison on a dictionary of ~128 000 English words (with 3+ characters), tested with 5 000 randomly generated misspellings (25% of substitutions, insertions, deletions, and swaps). All measurements are averaged over 5 trials. Tested on a Ryzen 9 365.

### Overall Accuracy and Speed

| Method               | Top‑1 (%) | Top‑3 (%) | Top‑5 (%) | Duration (s) | Build (s) | Size (MB) |
|----------------------|-----------|-----------|-----------|--------------|-----------|-----------|
| DPVS (CPU)           | 83.29% 🥇 | 92.72% 🥇| 94.81% 🥇 | 0.348s 🥈   | 24.742 🥉 | 92.95 🥈 |
| SymSpell             | 79.08%    | 90.65%    | 92.88%    | 0.169s 🥇    | 1.987 🥇 | 190.30 🥉 |
| Damerau‑Lev. BK‑Tree | 80.72% 🥈 | 92.11% 🥈| 94.71% 🥈 | 638.425s     | 5.385 🥈 | 38.37 🥇  |
| Jaro‑Winkler         | 80.56% 🥉 | 92.08% 🥉| 94.68% 🥉 | 422.227s     | N/A       | N/A       |
| RapidFuzz            | 80.05%    | 91.58%    | 94.36%    | 224.051s     | N/A       | N/A       |
| Norvig               | 79.36%    | 90.59%    | 92.84%    | 46.704s 🥉   | N/A       | N/A       |
| Levenshtein          | 70.70%    | 85.31%    | 89.01%    | 578.621s     | N/A       | N/A       |
| Bigram Jaccard       | 57.81%    | 75.44%    | 80.58%    | 1573.073s    | N/A       | N/A       |

### Top‑1 Accuracy by Typo Type

| Method               | Substitution | Insertion | Deletion | Transposition |
|----------------------|--------------|-----------|----------|---------------|
| DPVS                 | 72.3%        | 94.2%     | 67.3% 🥉 | 98.6% 🥇     |
| SymSpell             | 80.9%        | 95.7% 🥈  | 55.1%    | 84.2%         |
| Damerau‑Lev. BK‑Tree | 82.4% 🥇     | 93.5%     | 60.9%    | 85.7% 🥉     |
| Jaro‑Winkler         | 62.5%        | 94.6%     | 74.1% 🥈 | 90.2% 🥈     |
| RapidFuzz            | 71.1%        | 97.4% 🥇  | 79.9% 🥇| 71.7%         |
| Norvig               | 81.5% 🥉     | 95.3% 🥉 | 55.6%    | 84.5%         |
| Levenshtein          | 82.0% 🥈     | 94.3%     | 58.1%    | 48.7%         |
| Bigram Jaccard       | 55.2%        | 85.7%     | 55.1%    | 35.5%         |

### Top‑1 Accuracy by Error Count and Error Position

| Method               | 1‑Error | 2‑Errors | Prefix  | Middle  | Suffix  |
|----------------------|---------|----------|---------|---------|---------|
| DPVS                 | 87.9% 🥇| 65.3% 🥉| 81.6% 🥇| 89.0% 🥇| 73.8% 🥈|
| SymSpell             | 83.4%   | 62.2%    | 74.6% 🥉| 87.1%   | 67.7%   |
| Damerau‑Lev. BK‑Tree | 85.0% 🥈| 64.0%    | 75.6% 🥈| 88.5% 🥈| 70.5%   |
| Jaro‑Winkler         | 84.3% 🥉| 65.9% 🥈 | 72.2%   | 87.7%   | 74.9% 🥇|
| RapidFuzz            | 83.5%   | 66.4% 🥇 | 72.9%   | 88.0% 🥉| 71.5% 🥉|
| Norvig               | 83.7%   | 62.3%    | 74.4%   | 87.4%   | 68.6%   |
| Levenshtein          | 74.6%   | 55.4%    | 62.7%   | 78.6%   | 63.1%   |
| Bigram Jaccard       | 61.9%   | 41.7%    | 50.5%   | 65.6%   | 49.8%   |

---

## Installation

No pip package yet, the project is under active development. Import it directly as `import dpvs` after placing `dpvs.py` in your working directory.
Clone or download this repository, make sure you have Python 3.8 or newer, and install the required dependencies.

**Dependencies:**

- `faiss-cpu` (or `faiss-gpu` if you have an NVIDIA GPU and CUDA)
- `numpy`

Install everything with:

```bash
pip install faiss-cpu numpy
```

---

## Quick Start

The repository includes an `example.py` file you can run immediately. Below is a compact walk‑through.

```python
import dpvs

# Build a small vocabulary, you can use any list of strings
words = ["apple", "banana", "orange", "peach", "pineapple"]

# Create the index and build it
index = dpvs.VectorIndex().build(words)

# Look up 3 nearest neighbours for each fuzzy query
queries = ["aple", "bannana", "orng"]
results = index.lookup(queries, k=3)

for query, candidates in results:
    print(f"Candidates for '{query}':")
    for idx, distance in candidates:
        print(f"  → {words[idx]} (L1 distance: {distance:.4f})")

# Expected output:
#    Candidates for 'aple':
#        → apple (L1 distance: 2.8683)
#        → pineapple (L1 distance: 11.8561)
#        → peach (L1 distance: 12.5534)
#    Candidates for 'bannana':
#        → banana (L1 distance: 2.8639)
#        → orange (L1 distance: 20.7241)
#        → peach (L1 distance: 23.2924)
#    Candidates for 'orng':
#        → orange (L1 distance: 6.9839)
#        → banana (L1 distance: 19.1052)
#        → peach (L1 distance: 19.3234)
```

---

## Contributing

This is an early‑stage tool, and I’d love your ideas. Open an issue or pull request if you find a bug, have a feature request, or want to improve performance.

Please keep the code clean, the documentation human‑friendly, and the benchmarks honest.

---

## License

DPVS is provided under the MIT License. Use it, modify it, ship it in your product. Just retain the copyright notice.

If you build something interesting with it, I’d love to hear from you!
