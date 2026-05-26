import random
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from time import time
from spellchecker import SpellChecker
from tqdm import tqdm
import jellyfish
from rapidfuzz import fuzz
from pyxdameraulevenshtein import damerau_levenshtein_distance
from symspellpy import SymSpell, Verbosity
import pybktree
import dpvs
from unicodedata import normalize
import heapq
import json
from pympler import asizeof
import faiss

# ---------------------------
# TYPO GENERATORS
# ---------------------------

SUB_WEIGHT = 0.25
SWAP_WEIGHT = 0.25
DEL_WEIGHT = 0.25
INS_WEIGHT = 0.25

def get_pos_label(i, length):
    if length <= 3:
        return "middle"
    if i == 0 or i == 1:
        return "prefix"
    elif i >= length - 2:
        return "suffix"
    else:
        return "middle"

def typo_substitution(word):
    if len(word) < 2:
        return word, "middle"
    i = random.randint(0, len(word)-1)
    c = random.choice("abcdefghijklmnopqrstuvwxyz".replace(word[i], ""))
    return word[:i] + c + word[i+1:], get_pos_label(i, len(word))


def typo_swap(word):
    if len(word) < 2:
        return word, "middle"
    i = random.randint(0, len(word)-2)
    lst = list(word)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst), get_pos_label(i, len(word))


def typo_deletion(word):
    if len(word) < 2:
        return word, "middle"
    i = random.randint(0, len(word)-1)
    return word[:i] + word[i+1:], get_pos_label(i, len(word))


def typo_insertion(word):
    i = random.randint(0, len(word))
    c = random.choice("abcdefghijklmnopqrstuvwxyz")
    return word[:i] + c + word[i:], get_pos_label(i, len(word))


def generate_typos(vocab, n=5000, difficulty="mixed"):
    # realistic weights: Sub is most common, then Ins/Del, then Swap
    # You can adjust these weights
    types = [
        ("substitution", typo_substitution),
        ("swap", typo_swap),
        ("deletion", typo_deletion),
        ("insertion", typo_insertion)
    ]
    weights = [SUB_WEIGHT, SWAP_WEIGHT, DEL_WEIGHT, INS_WEIGHT]
    
    test_cases = []

    for _ in range(n):
        w = random.choice(vocab)
        while len(w) < 4:
            w = random.choice(vocab)

        t_name, t_func = random.choices(types, weights=weights, k=1)[0]
        
        num_edits = 1
        if difficulty == "2-edit":
            num_edits = 2
        elif difficulty == "mixed":
            num_edits = random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]
        
        current_w = w
        pos_label = "middle"
        for _ in range(num_edits):
            current_w, pos_label = t_func(current_w)

        test_cases.append({
            "query": current_w,
            "target": w,
            "error_type": t_name,
            "error_pos": pos_label,
            "edits": num_edits
        })

    return test_cases


# ---------------------------
# BASELINE METHODS
# ---------------------------

def candidates_linear_scan(query, vocab, distance_func, k=5):
    """Generic linear scan using a heap to keep top-k."""
    heap = []
    for w in vocab:
        dist = distance_func(query, w)
        # Push negative distance for max-heap behaviour (we want smallest distances)
        heapq.heappush(heap, (-dist, w))
        if len(heap) > k:
            heapq.heappop(heap)
    # Extract words sorted by distance (closest first)
    return [w for _, w in sorted(heap, key=lambda x: -x[0])]

def candidates_levenshtein(query, vocab, k=5):
    return candidates_linear_scan(query, vocab, jellyfish.levenshtein_distance, k)


def candidates_damerau_levenshtein(query, vocab, k=5):
    return candidates_linear_scan(query, vocab, damerau_levenshtein_distance, k)


def candidates_jaro_winkler(query, vocab, k=5):
    # Jaro-Winkler similarity: higher is better, so negate for heap
    def jw_dist(a, b):
        return -jellyfish.jaro_winkler_similarity(a, b)
    return candidates_linear_scan(query, vocab, jw_dist, k)


def candidates_rapidfuzz(query, vocab, k=5):
    def rf_dist(a, b):
        return -fuzz.ratio(a, b)  # higher ratio = more similar, so negate
    return candidates_linear_scan(query, vocab, rf_dist, k)


def candidates_jaccard(query, vocab, k=5):
    def jaccard_dist(a, b):
        set_a = set(a[i:i+2] for i in range(len(a)-1)) if len(a) > 1 else {a}
        set_b = set(b[i:i+2] for i in range(len(b)-1)) if len(b) > 1 else {b}
        intersect = set_a.intersection(set_b)
        union = set_a.union(set_b)
        if not union:
            return 1.0
        return -len(intersect) / len(union)  # negate similarity for heap
    return candidates_linear_scan(query, vocab, jaccard_dist, k)


def candidates_symspell(query, vocab, symspell_instance, k=5):
    """SymSpell: use its built-in lookup, return top-k candidates."""
    suggestions = symspell_instance.lookup(query, Verbosity.CLOSEST, max_edit_distance=2)
    # suggestions are already sorted by (distance, frequency)
    return [s.term for s in suggestions[:k]]


def candidates_bktree(query, vocab, bktree_instance, k=5):
    """BK-Tree: find words within edit distance 2, sort by distance."""
    results = bktree_instance.find(query, 2)
    # results is a list of (distance, word); sort by distance
    results.sort(key=lambda x: x[0])
    return [word for dist, word in results[:k]]


def candidates_dpvs_batch(queries, vocab, dpvs_instance, k=5):
    """DPVS batched lookup. Returns list of candidate lists."""
    results = dpvs_instance.lookup(queries, k)
    return [[vocab[idx] for idx, dist in res[1]] for res in results]

def candidates_norvig(query, vocab, freq_dict, k=5):
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

# ---------------------------
# EVALUATION
# ---------------------------

def evaluate_accuracy(method_func, name, test_cases, vocab, args=[], trial_info="", is_batched=False):
    top1 = 0
    top3 = 0
    top5 = 0
    total = len(test_cases)
    
    # Stratified metrics
    stats = {
        "error_type": {"substitution": {"count": 0, "top1": 0}, "swap": {"count": 0, "top1": 0}, "deletion": {"count": 0, "top1": 0}, "insertion": {"count": 0, "top1": 0}},
        "error_pos": {"prefix": {"count": 0, "top1": 0}, "middle": {"count": 0, "top1": 0}, "suffix": {"count": 0, "top1": 0}},
        "edits": {1: {"count": 0, "top1": 0}, 2: {"count": 0, "top1": 0}}
    }

    t0 = time()

    desc_str = f"{trial_info} - {name}" if trial_info else name

    if is_batched:
        queries = [tc["query"] for tc in test_cases]
        all_preds = method_func(queries, vocab, *args)

    for i, tc in enumerate(tqdm(test_cases, total=total, desc=desc_str, leave=False)):
        target = tc["target"]
        
        if is_batched:
            preds = all_preds[i]
        else:
            q = tc["query"]
            preds = method_func(q, vocab, *args) if args else method_func(q, vocab)

        is_top1 = False
        if target in preds[:1]:
            top1 += 1
            is_top1 = True
        if target in preds[:3]:
            top3 += 1
        if target in preds[:5]:
            top5 += 1
            
        # Update stratified stats
        e_type = tc["error_type"]
        e_pos = tc["error_pos"]
        e_edits = tc["edits"]
        
        if e_type in stats["error_type"]:
            stats["error_type"][e_type]["count"] += 1
            if is_top1: stats["error_type"][e_type]["top1"] += 1
            
        if e_pos in stats["error_pos"]:
            stats["error_pos"][e_pos]["count"] += 1
            if is_top1: stats["error_pos"][e_pos]["top1"] += 1
            
        if e_edits in stats["edits"]:
            stats["edits"][e_edits]["count"] += 1
            if is_top1: stats["edits"][e_edits]["top1"] += 1

    t1 = time()
    duration = t1 - t0

    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "time_sec": duration,
        "iters_sec": total / duration if duration > 0 else 0.0,
        "stats": stats
    }


# ---------------------------
# RUN TEST
# ---------------------------

def run_benchmark(freq_dict, n_trials, n_per_trial, seed=0, save_to_file=False):
    vocab = list(freq_dict.keys())

    #Build SymSpell dictionary and measure build time (preprocessing)
    print("\nInitializing SymSpell dictionary (preprocessing)...")
    t0_build = time()
    
    symspell_instance = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for w in vocab:
        freq = freq_dict.get(w, 1)
        symspell_instance.create_dictionary_entry(w, max(1, freq)) # Ensure at least a count of 1
    
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

    print("\nBuilding BK-Tree index (preprocessing)...")
    t0_bktree = time()
    bktree_instance = pybktree.BKTree(damerau_levenshtein_distance, vocab)
    t1_bktree = time()
    bktree_build_time = t1_bktree - t0_bktree
    bktree_size = asizeof.asizeof(bktree_instance) / (1024 * 1024)

    # Define methods to benchmark
    methods = [
        (candidates_dpvs_batch, "DPVS", [dpvs_instance], True),
        (candidates_symspell, "SymSpell", [symspell_instance], False),
        (candidates_norvig, "Norvig", [freq_dict], False),
        (candidates_bktree, "Damerau BK-Tree", [bktree_instance], False),
        #(candidates_damerau_levenshtein, "Damerau-Levenshtein", [], False),    # Too slow for large vocabularies, replaced with BK-Tree which uses the same distance but is indexed
        (candidates_levenshtein, "Levenshtein", [], False),
        (candidates_jaro_winkler, "Jaro-Winkler", [], False),
        (candidates_rapidfuzz, "RapidFuzz", [], False),
        #(candidates_jaccard, "Bigram Jaccard", [], False),                     # Too slow for large vocabularies and generally performs poorly
    ]

    results = {name: {"top1": [], "top3": [], "top5": [], "time_sec": [], "iters_sec": [], "build_time": [], "build_size": [], "stats": []} for _, name, _, _ in methods}

    print("\nStarting Benchmark (", n_trials, "trials,", n_per_trial, "queries each)")

    for tr in range(n_trials):
        random.seed(seed + tr)
        np.random.seed(seed + tr)

        test_cases = generate_typos(vocab, n_per_trial, difficulty="mixed")
        trial_str = f"Trial {tr + 1}/{n_trials}"

        for func, name, args, is_batched in methods:
            res = evaluate_accuracy(func, name, test_cases, vocab, args, trial_info=trial_str, is_batched=is_batched)
            for k in ("top1", "top3", "top5", "time_sec", "iters_sec"):
                results[name][k].append(res[k])
            results[name]["stats"].append(res["stats"])
            # record build time relevant to method
            if name.startswith("SymSpell"):
                results[name]["build_time"].append(symspell_build_time)
                results[name]["build_size"].append(symspell_size)
            elif name.startswith("DPVS"):
                results[name]["build_time"].append(dpvs_build_time)
                results[name]["build_size"].append(dpvs_size)
            elif name.startswith("Damerau BK-Tree"):
                results[name]["build_time"].append(bktree_build_time)
                results[name]["build_size"].append(bktree_size)
            else:
                results[name]["build_time"].append(0.0)
                results[name]["build_size"].append(0.0)

    # Print aggregated mean +/- std, include symspell build time separately
    print("\n" + "=" * 135)
    print(f"{'Method':<20} | {'Top-1 (%)':<15} | {'Top-3 (%)':<15} | {'Top-5 (%)':<15} | {'Duration (s)':<15} | {'Iter/s':<15} | {'Build (s)':<10} | {'Size (MB)':<10}")
    print("-" * 135)

    def mean_std(arr):
        if not arr: return 0.0, 0.0
        a = np.asarray(arr)
        return a.mean(), a.std()

    for _, name, _, is_batched in methods:
        t1_mean, t1_std = mean_std(results[name]["top1"]) ; t1_mean *= 100 ; t1_std *= 100
        t3_mean, t3_std = mean_std(results[name]["top3"]) ; t3_mean *= 100 ; t3_std *= 100
        t5_mean, t5_std = mean_std(results[name]["top5"]) ; t5_mean *= 100 ; t5_std *= 100
        time_mean, time_std = mean_std(results[name]["time_sec"])
        iters_mean, iters_std = mean_std(results[name]["iters_sec"])
        build_mean, _ = mean_std(results[name]["build_time"])
        size_mean, _ = mean_std(results[name]["build_size"])

        print(f"{name:<20} | {f'{t1_mean:>5.2f}% ±{t1_std:4.2f}':<15} | {f'{t3_mean:>5.2f}% ±{t3_std:4.2f}':<15} | {f'{t5_mean:>5.2f}% ±{t5_std:4.2f}':<15} | {f'{time_mean:>7.3f}s ±{time_std:4.3f}':<15} | {f'{iters_mean:>8.1f} ±{iters_std:<5.1f}' if is_batched else f'{iters_mean:>7.1f} ±{iters_std:<4.1f}':<15} | {f'{build_mean:>8.3f}s':<10} | {f'{size_mean:>8.2f}':<10}")
    print("=" * 135)

    print("\n=== Diagnostic Top-1 Accuracy Breakdown " + "=" * 83)
    
    # helper for aggregating stats across trials
    def agg_stats(name, category, key):
        t_counts = 0
        t_top1 = 0
        for trial_stat in results[name]["stats"]:
            t_counts += trial_stat[category][key]["count"]
            t_top1 += trial_stat[category][key]["top1"]
        return (t_top1 / t_counts * 100) if t_counts > 0 else 0.0

    print(f"{'Method':<20} | {'Sub':<7} {'Ins':<7} {'Del':<7} {'Swap':<7} | {'Prefx':<7} {'Middl':<7} {'Suffx':<7} | {'1-Edit':<7} {'2-Edits':<7}")
    print("=" * 123)
    for _, name, _, _ in methods:
        sub_acc = agg_stats(name, "error_type", "substitution")
        ins_acc = agg_stats(name, "error_type", "insertion")
        del_acc = agg_stats(name, "error_type", "deletion")
        swap_acc = agg_stats(name, "error_type", "swap")
        
        pref_acc = agg_stats(name, "error_pos", "prefix")
        mid_acc = agg_stats(name, "error_pos", "middle")
        suf_acc = agg_stats(name, "error_pos", "suffix")
        
        e1_acc = agg_stats(name, "edits", 1)
        e2_acc = agg_stats(name, "edits", 2)
        
        print(f"{name:<20} | {f'{sub_acc:>5.1f}%':<7} {f'{ins_acc:>5.1f}%':<7} {f'{del_acc:>5.1f}%':<7} {f'{swap_acc:>5.1f}%':<7} | {f'{pref_acc:>5.1f}%':<7} {f'{mid_acc:>5.1f}%':<7} {f'{suf_acc:>5.1f}%':<7} | {f'{e1_acc:>5.1f}%':<7} {f'{e2_acc:>5.1f}%':<7}")
    print("=" * 123)

    if save_to_file:
        with open("benchmark_raw_data.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print("\nSaved raw benchmark data to benchmark_raw_data.json")

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    freq_dict = SpellChecker().word_frequency.dictionary
    
    print("Filtering and normalizing vocabulary...")
    
    # Normalize words to NFD form to ensure consistent character representation, and filter out very short words
    filtered_dict = {
        normalize('NFD', w): freq 
        for w, freq in freq_dict.items()
        if len(w) > 2
    }
    
    print(f"Using {len(filtered_dict)}/{len(freq_dict)} alphabetic words for the benchmark")
    
    run_benchmark(filtered_dict, n_trials=5, n_per_trial=5_000, seed=0, save_to_file=False)