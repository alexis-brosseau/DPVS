# DPVS: Deterministic Positional Vectorization of Strings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast approximate string matching library** that turns words into compact vectors so you can find the closest match, even when the query is riddled with typos! It’s like fuzzy matching on steroids: sub‑millisecond lookup times and linear memory scaling.

## Why DPVS?

I built DPVS because I needed to search through a large vocabulary with queries that often contained swapped letters, missing characters, and random insertions. Traditional edit‑distance algorithms were accurate but far too slow for real‑time use. Faster alternatives like SymSpell were quick, but they fell apart on transpositions ("lorem" → "olrem"). Their memory footprint also exploded with higher error tolerances!

DPVS solves these problems:

- **Speed:** FAISS‑powered HNSW index gives you ~14 000 queries per second **on a laptop CPU** (and even faster with a GPU!).
- **Transpositions Accuracy:** It correctly catches 99 % of transpositions errors, beating all other algorithms by a wide margin.
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

Here’s a quick comparison on a dictionary of ~160 000 English words (with 3+ characters), tested with 5 000 randomly generated misspellings (25% of substitutions, insertions, deletions, and transposition). All measurements are averaged over 5 trials. Tested on a Ryzen 9 365.

### Overall Accuracy and Speed

| Method               | Top‑1 (%) | Top‑3 (%) | Top‑5 (%) | Duration (s) | Build (s) | Size (MB) |
|----------------------|-----------|-----------|-----------|--------------|-----------|-----------|
| DPVS (CPU)           | 82.79% 🥇 | 92.80% 🥇| 94.94% 🥇 | 0.438s 🥈   | 27.126    | 104.81    |
| SymSpell             | 78.24%    | 90.11%    | 92.16%    | 0.164s 🥇    | 1.987     | 190.31    |
| RapidFuzz            | 79.96% 🥈 | 91.70%    | 94.42% 🥉| 81.064s     | N/A       | N/A       |
| Jaro‑Winkler         | 79.24% 🥉 | 92.02% 🥈| 94.56% 🥈 | 79.911s      | N/A       | N/A       |
| Damerau-Levenshtein  | 78.24%    | 92.11% 🥉| 93.92%     | 614.175s     | N/A       | N/A  |
| Levenshtein          | 68.70%    | 84.27%    | 88.22%    | 75.773s     | N/A       | N/A       |
| Norvig               | 79.36%    | 90.59%    | 91.28%    | 58.657s 🥉   | N/A       | N/A       |

### Top‑1 Accuracy by Typo Type

| Method               | Substitution | Insertion | Deletion | Transposition |
|----------------------|--------------|-----------|----------|---------------|
| DPVS                 | 71.8%        | 94.2%     | 65.8% 🥉 | 99.0% 🥇     |
| SymSpell             | 81.5% 🥉     | 95.5% 🥉  | 51.7%    | 84.6% 🥉     |
| RapidFuzz            | 72.5%        | 97.9% 🥇  | 78.4% 🥇| 71.1%         |
| Jaro‑Winkler         | 60.1%        | 95.2%     | 72.0% 🥈 | 89.1% 🥈     |
| Damerau-Levenshtein  | 82.3% 🥇     | 94.1%     | 52.6%    | 84.4%         |
| Levenshtein          | 82.3% 🥈     | 94.2%     | 53.0%    | 46.2%         |
| Norvig               | 80.6%        | 95.7% 🥈  | 52.8%    | 84.0%         |

### Top‑1 Accuracy by Error Count and Error Position

| Method               | 1‑Error | 2‑Errors | Prefix  | Middle  | Suffix  |
|----------------------|---------|----------|---------|---------|---------|
| DPVS                 | 87.2% 🥇| 65.5% 🥉| 81.1% 🥇| 88.1% 🥇| 73.1% 🥇|
| SymSpell             | 82.3%   | 62.4%    | 74.5% 🥉| 85.8%   | 65.8%   |
| RapidFuzz            | 83.3% 🥈| 66.9% 🥇 | 73.4%   | 87.0% 🥈| 71.5% 🥉|
| Jaro‑Winkler         | 82.8% 🥉| 65.4% 🥈 | 71.5%   | 86.5% 🥉| 71.6% 🥈|
| Damerau-Levenshtein  | 82.3%   | 62.3%    | 75.3% 🥈| 86.1%   | 64.3%   |
| Levenshtein          | 72.3%   | 54.7%    | 62.7%   | 76.8%   | 58.0%   |
| Norvig               | 82.3%   | 62.3%    | 74.4%   | 85.8%   | 65.7%   |

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

The repository includes examples files that you can run immediately under `/examples`. Note that `pdvs.py` must be in the same directory. Below is a compact walk‑through.

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
