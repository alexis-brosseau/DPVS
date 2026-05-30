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

### Real-World Human Error Benchmark (Birkbeck Spelling Error Corpus)
Here’s a comparison on a dictionary of ~160 000 English words, tested with the Birkbeck Spelling Error Corpus. This dataset consists of non-synthetic human misspellings including heavy phonetic mutations, dysgraphia and multi-error handwriting slips. Tested on a Ryzen 9 365.

| Method | Top-1 (%) | Top-5 (%) | Top-10 (%) | Top-25 (%) | Top-100 (%) | Duration (s) | Build (s) | Size (MB) |
|--------|-----------|-----------|------------|------------|-------------|--------------|-----------|-----------|
| DPVS                 | 29.24%     | 46.70%     | 53.25%     | 61.75%     | 71.26%      | 2.741s 🥇      | 22.074s    | 110.06     |
| SymSpell             | 34.06% 🥇   | 48.92% 🥉   | 51.94%     | 54.58%     | 57.70%      | 12.596s 🥈     | 37.902s    | 3568.23    |
| RapidFuzz            | 32.65% 🥉   | 51.74% 🥇   | 58.54% 🥇   | 66.56% 🥇   | 76.67% 🥇    | 412.887s 🥉    | N/A        | N/A        |
| Jaro-Winkler         | 30.27%     | 50.72% 🥈   | 57.66% 🥈   | 65.43% 🥈   | 75.86% 🥈    | 518.326s      | N/A        | N/A        |
| Damerau-Levenshtein  | 29.20%     | 48.10%     | 55.56% 🥉   | 63.92% 🥉   | 73.18% 🥉    | 3431.185s     | N/A        | N/A        |
| Levenshtein          | 28.10%     | 46.73%     | 54.20%     | 62.64%     | 72.35%      | 463.465s      | N/A        | N/A        |
| Norvig               | 33.77% 🥈   | 40.80%     | 41.33%     | 41.48%     | 41.48%      | 746.374s      | N/A        | N/A        |

### Synthetic Dataset
Here’s a quick comparison on the same dictionary of ~160 000 English words (with 4+ characters), tested with 5 000 randomly generated misspellings (25% of substitutions, insertions, deletions, and transposition). All measurements are averaged over 5 trials. Tested on a Ryzen 9 365.

#### Overall Accuracy and Speed

| Method | Top-1 (%) | Top-3 (%) | Top-5 (%) | Duration (s) | Build (s) | Size (MB) |
|---|---|---|---|---|---|---|
| DPVS                 | 82.58% 🥇   | 92.65% 🥇   | 95.09% 🥇   | 0.255s 🥈   | 21.383s  | 108.95    |
| SymSpell             | 78.53%     | 90.76%     | 92.94%     | 0.170s 🥇   | 1.982s   | 190.17    |
| RapidFuzz            | 80.10% 🥈   | 91.85%     | 94.72% 🥉   | 55.423s    | N/A      | N/A       |
| Jaro-Winkler         | 79.68% 🥉   | 92.33% 🥈   | 94.76% 🥈   | 71.860s    | N/A      | N/A       |
| Damerau-Levenshtein  | 79.06%     | 91.95% 🥉   | 94.47%     | 528.623s   | N/A      | N/A       |
| Levenshtein          | 69.53%     | 85.02%     | 88.98%     | 62.813s    | N/A      | N/A       |
| Norvig               | 78.40%     | 89.90%     | 92.10%     | 44.230s 🥉  | N/A      | N/A       |

#### Top‑1 Accuracy by Error Type

| Method | Substitution | Insertion | Deletion | Transposition |
|----------------------|--------------|-----------|----------|---------------|
| DPVS                 |   69.3%     |   94.1%   |   67.2% 🥉 |   99.0% 🥇    |
| SymSpell             |   80.8% 🥉    |   94.9%   |   53.9%  |   84.4%     |
| RapidFuzz            |   71.3%     |   97.8% 🥇  |   79.8% 🥇 |   71.0%     |
| Jaro-Winkler         |   60.2%     |   95.3% 🥈  |   73.5% 🥈 |   88.9% 🥈    |
| Damerau-Levenshtein  |   81.7% 🥈    |   94.5%   |   55.3%  |   84.7% 🥉    |
| Levenshtein          |   81.8% 🥇    |   94.5%   |   55.6%  |   46.4%     |
| Norvig               |   79.9%     |   95.1% 🥉  |   54.6%  |   83.9%     |

#### Top‑1 Accuracy by Error Count and Error Position

| Method | 1‑Error | 2‑Errors | Prefix | Middle | Suffix |
|----------------------|---------|----------|--------|--------|--------|
| DPVS                 |   87.3% 🥇 |    63.9% 🥇 |   81.1% 🥇 |   86.9% 🥈|   74.5% 🥇 |
| SymSpell             |   82.7% |    61.7% |         74.4% 🥉|   85.9% |   66.5% |
| RapidFuzz            |   83.8% 🥈|    65.5% 🥉 |   73.8% |   86.7% 🥉|   72.1% 🥈 |
| Jaro-Winkler         |   83.1% 🥉 |    65.9% 🥈 |   71.8% |   87.1% 🥇|   71.6% 🥉 |
| Damerau-Levenshtein  |   83.0% |        62.8% |   76.0% 🥈|   86.3% |   66.3% |
| Levenshtein          |   73.3% |        54.6% |   63.9% |   76.5% |   60.1% |
| Norvig               |   82.6% |        61.5% |   74.1% |   85.9% |   66.4% |

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
