import dpvs

# Build a small vocabulary – you can use any list of strings
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