import dpvs

# Build a small list of words
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