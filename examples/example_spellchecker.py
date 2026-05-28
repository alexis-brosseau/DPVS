import dpvs
from spellchecker import SpellChecker

if __name__ == "__main__":
    # Get words from pyspellchecker's frequency dictionary
    words = list(SpellChecker().word_frequency.dictionary.keys())
    typos = ["teh", "recieve", "definately", "occured", "publically"]
    index = dpvs.VectorIndex()
    
    print(f"Building vector index for {len(words)} words...")
    index.build(words)
    
    print(f"Looking up candidates...\n")
    results = index.lookup(typos, k=3)
    
    for query, candidates in results:
        print(f"Candidates for '{query}': {[f'{candidate}: {dist:.2f}' for candidate, dist in candidates]}")