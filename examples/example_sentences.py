import dpvs

# Build a small list of sentences / product names
products = [
    "iPhone 17 Pro Max",
    "iPhone 17 Pro",
    "iPhone Air",
    "iPhone 17",
    "iPhone 17e",
    "iPhone 16e",
    "iPhone 16 Pro Max",
    "iPhone 16 Pro",
    "iPhone 16 Plus",
    "iPhone 16",
    "iPhone 15 Pro Max",
    "iPhone 15 Pro",
    "iPhone 15 Plus",
    "iPhone 15",
    "iPhone 14 Pro Max",
    "iPhone 14 Pro",
    "iPhone 14 Plus",
    "iPhone 14",
    "iPhone SE 3rd generation",
    "iPhone 13 Pro Max",
    "iPhone 13 Pro",
    "iPhone 13",
    "iPhone 13 mini",
    "iPhone 12 Pro Max",
    "iPhone 12 Pro",
    "iPhone 12",
    "iPhone 12 mini",
    "iPhone SE 2nd generation",
    "iPhone 11 Pro Max",
    "iPhone 11 Pro",
    "iPhone 11",
    "iPhone XS Max",
    "iPhone XS",
    "iPhone XR",
    "iPhone X",
    "iPhone 8 Plus",
    "iPhone 8",
    "iPhone 7 Plus",
    "iPhone 7",
    "iPhone SE 1st generation",
    "iPhone 6s Plus",
    "iPhone 6s",
    "iPhone 6 Plus",
    "iPhone 6",
    "iPhone 5s",
    "iPhone 5c",
    "iPhone 5",
    "iPhone 4s",
    "iPhone 4",
    "iPhone 3GS",
    "iPhone 3G",
    "iPhone 1st generation"
]


# Create the index with sentence-level vectorization and build it
index = dpvs.VectorIndex(dpvs.IndexType.SENTENCE).build(products)

# Look up 3 nearest neighbours for each fuzzy query
queries = [
    "17 pro",
    "iphn 14",
    "SE 3rd gen",
    "6s",
    "iphon mini 12",
    "14 iphone max pro",
]
        
results = index.lookup(queries, k=3)

for query, candidates in results:
    print(f"Candidates for '{query}':")
    for idx, distance in candidates:
        print(f"  → {products[idx]} (Dist: {distance:.4f})")