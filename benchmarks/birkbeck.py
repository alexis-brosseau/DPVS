import os
import sys
from time import time
from unicodedata import normalize
from tqdm import tqdm
from spellchecker import SpellChecker
from symspellpy import SymSpell
from pympler import asizeof
import dpvs
import numpy as np
import matplotlib.pyplot as plt
from time import time
from spellchecker import SpellChecker
from tqdm import tqdm
import jellyfish
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein, DamerauLevenshtein, JaroWinkler
from symspellpy import SymSpell, Verbosity
import pybktree
import dpvs
from unicodedata import normalize
import json
from pympler import asizeof

# ---------------------------
# BASELINE METHODS
# ---------------------------

def candidates_levenshtein(query, vocab, k=100):
    results = process.extract(query, vocab, scorer=Levenshtein.distance, limit=k)
    return [match[0] for match in results]

def candidates_damerau_levenshtein(query, vocab, k=100):
    results = process.extract(query, vocab, scorer=DamerauLevenshtein.distance, limit=k)
    return [match[0] for match in results]

def candidates_jaro_winkler(query, vocab, k=100):
    results = process.extract(query, vocab, scorer=JaroWinkler.similarity, limit=k)
    return [match[0] for match in results]

def candidates_rapidfuzz(query, vocab, k=100):
    results = process.extract(query, vocab, scorer=fuzz.ratio, limit=k)
    return [match[0] for match in results]

def candidates_symspell(query, vocab, symspell_instance, k=100):
    """SymSpell: use its built-in lookup, return top-k candidates."""
    suggestions = symspell_instance.lookup(query, Verbosity.ALL, max_edit_distance=2)
    # suggestions are already sorted by (distance, frequency)
    return [s.term for s in suggestions[:k]]

def candidates_bktree(query, vocab, bktree_instance, k=100):
    """BK-Tree: find words within edit distance 2, sort by distance."""
    results = bktree_instance.find(query, 2)
    # results is a list of (distance, word); sort by distance
    results.sort(key=lambda x: x[0])
    return [word for dist, word in results[:k]]

def candidates_dpvs_batch(queries, vocab, dpvs_instance, k=100):
    """DPVS batched lookup. Returns list of candidate lists."""
    results = dpvs_instance.lookup(queries, k)
    return [[word for word, dist in res[1]] for res in results]

def candidates_norvig(query, vocab, freq_dict, k=100):
    """Norvig's spelling corrector candidate generation."""
    def edits1(word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word): 
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))
        
    def known(words): 
        return set(w for w in words if w in freq_dict)
        
    cands = known([query]) or known(edits1(query)) or known(edits2(query)) or {query}
    return sorted(cands, key=lambda w: freq_dict.get(w, 0), reverse=True)[:k]


def load_birkbeck_dataset(filepath):
    test_cases = []
    targets = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) == 2:
                # birkbeck contains entries like "a_bit: abit", replacing underscores with spaces
                target = parts[0].strip().replace("_", " ").lower()
                targets.add(target)
                queries = parts[1].strip().split()
                for q in queries:
                    q = q.replace("_", " ").lower()
                    test_cases.append({"query": q, "target": target})
    return test_cases, targets

def evaluate_simple(method_func, name, test_cases, vocab, args=[], is_batched=False):
    top1, top5, top10, top25, top100 = 0, 0, 0, 0, 0
    total = len(test_cases)
    
    t0 = time()
    
    if is_batched:
        print(f"Running {name} in batch mode...", end="\r", flush=True)
        queries = [tc["query"] for tc in test_cases]
        all_preds = method_func(queries, vocab, *args)
        
        for tc, preds in zip(test_cases, all_preds):
            target = tc["target"]
            if target in preds[:1]: top1 += 1
            if target in preds[:5]: top5 += 1
            if target in preds[:10]: top10 += 1
            if target in preds[:25]: top25 += 1
            if target in preds[:100]: top100 += 1
    else:
        for tc in tqdm(test_cases, desc=name, leave=False):
            target = tc["target"]
            preds = method_func(tc["query"], vocab, *args)
            if target in preds[:1]: top1 += 1
            if target in preds[:5]: top5 += 1
            if target in preds[:10]: top10 += 1
            if target in preds[:25]: top25 += 1
            if target in preds[:100]: top100 += 1
            
    t1 = time()
    duration = t1 - t0
    
    return {
        "top1": top1 / total,
        "top5": top5 / total,
        "top10": top10 / total,
        "top25": top25 / total,
        "top100": top100 / total,
        "time_sec": duration,
        "iters_sec": total / duration if duration > 0 else 0.0
    }

def run_birkbeck_benchmark():
    data_path = os.path.join(os.path.dirname(__file__), "birkbeck.txt")
    print(f"Loading dataset from {data_path}...")
    test_cases, birkbeck_targets = load_birkbeck_dataset(data_path)
    print(f"Loaded {len(test_cases)} test cases from birkbeck dataset.")

    print("Filtering and normalizing vocabulary...")
    freq_dict = SpellChecker().word_frequency.dictionary
    
    filtered_dict = {}
    for w, freq in freq_dict.items():
        norm_w = normalize('NFD', w).lower()
        filtered_dict[norm_w] = freq
        
    # Ensure all birkbeck targets are securely in our vocabulary so it's a fair test for all algorithms
    for t in birkbeck_targets:
        norm_t = normalize('NFD', t).lower()
        if norm_t not in filtered_dict:
            filtered_dict[norm_t] = 100  # Give it an arbitrary frequency
            
    vocab = list(filtered_dict.keys())
    print(f"Using {len(vocab)} words for the benchmark vocabulary.")

    #Build SymSpell dictionary and measure build time
    print("\nInitializing SymSpell dictionary (preprocessing)...")
    t0_build = time()
    symspell_instance = SymSpell(max_dictionary_edit_distance=4, prefix_length=12)
    for w in vocab:
        freq = filtered_dict.get(w, 1)
        symspell_instance.create_dictionary_entry(w, max(1, freq))
    t1_build = time()
    symspell_build_time = t1_build - t0_build
    symspell_size = asizeof.asizeof(symspell_instance) / (1024 * 1024)

    # Build DPVS index
    print("\nBuilding DPVS index (preprocessing)...")
    t0_dpvs = time()
    dpvs_instance = dpvs.VectorIndex().build(vocab)
    t1_dpvs = time()
    dpvs_build_time = t1_dpvs - t0_dpvs
    dpvs_size = asizeof.asizeof(dpvs_instance) / (1024 * 1024)

    # Define methods to benchmark
    methods = [
        (candidates_dpvs_batch, "DPVS", [dpvs_instance], True),
        (candidates_symspell, "SymSpell", [symspell_instance], False),
        (candidates_rapidfuzz, "RapidFuzz", [], False),
        (candidates_jaro_winkler, "Jaro-Winkler", [], False),
        (candidates_damerau_levenshtein, "Damerau-Levenshtein", [], False),
        (candidates_levenshtein, "Levenshtein", [], False),
        (candidates_norvig, "Norvig", [filtered_dict], False),
    ]

    print("\nStarting Benchmark on birkbeck dataset...")
    results = []
    for func, name, args, is_batched in methods:
        res = evaluate_simple(func, name, test_cases, vocab, args, is_batched)
        res["name"] = name
        
        if name == "SymSpell":
            res["build_time"] = symspell_build_time
            res["build_size"] = symspell_size
        elif name == "DPVS":
            res["build_time"] = dpvs_build_time
            res["build_size"] = dpvs_size
        else:
            res["build_time"] = 0.0
            res["build_size"] = 0.0
            
        results.append(res)
        
    print("\n" + "=" * 175)
    print(f"{'Method':<20} | {'Top-1 (%)':<15} | {'Top-5 (%)':<15} | {'Top-10 (%)':<15} | {'Top-25 (%)':<15} | {'Top-100 (%)':<15} | {'Duration (s)':<15} | {'Iter/s':<10} | {'Build (s)':<10} | {'Size (MB)':<10}")
    print("-" * 175)

    for r in results:
        t1 = r['top1'] * 100
        t5 = r['top5'] * 100
        t10 = r['top10'] * 100
        t25 = r['top25'] * 100
        t100 = r['top100'] * 100
        d = r['time_sec']
        it = r['iters_sec']
        bt = r['build_time']
        bs = r['build_size']
        
        print(f"{r['name']:<20} | {t1:>5.2f}%          | {t5:>5.2f}%          | {t10:>5.2f}%          | {t25:>5.2f}%          | {t100:>5.2f}%          | {d:>7.3f}s        | {it:>8.1f} | {bt:>8.3f}s    | {bs:>8.2f}")
    print("=" * 175)

if __name__ == "__main__":
    run_birkbeck_benchmark()
