import pickle
from enum import Enum
import numpy as np
import faiss        # Use `pip install faiss-cpu` or `pip install faiss-gpu` depending on your system

class IndexType(Enum):
    WORD = 1
    SENTENCE = 2

class VectorIndex:
    """
    This class handles the vectorization of strings and stores them 
    in an indexing structure (FAISS HNSW) for efficient similarity search
    using Deterministic Positional Vectorization of Strings (DPVS).
    """

    def __init__(self, index_type: IndexType=IndexType.WORD, chars: str="abcdefghijklmnopqrstuvwxyz0123456789-̧ ' ", ef_construction: int=200, M: int=32, ef: int=50):
        """
        Initialize the DPVS model and its underlying FAISS index parameters.

        Args:
            chars (str): A string containing all valid characters for vectorization.
            ef_construction (int, optional): The depth of the search during index construction for FAISS HNSW. Defaults to 200.
            M (int, optional): The number of bi-directional links created for every new element during HNSW index construction. Defaults to 32.
            ef (int, optional): The depth of the search for FAISS HNSW. Defaults to 50.
        """
        self.type = index_type
        self.vectorize = self._word_vector if index_type == IndexType.WORD else self._sentence_vector
        
        self.chars = chars
        self.chars_len = len(chars)
        self.char_idx = {c: i for i, c in enumerate(chars)}
        
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        
        self.vectors = None
        self.index = None
            
    def build(self, entries: list[str]):
        """
        Build the FAISS index from the provided entries.

        Args:
            entries (list[str]): A list of strings to vectorize and index.
        """
        self.entries = entries
        self.vectors = self._build_entries_vectors(entries)
        self.index = self._build_faiss_index(metric=faiss.METRIC_L1, ef_construction=self.ef_construction, M=self.M, ef=self.ef)
        
        return self

    def save(self, filepath: str="dpvs_index.pkl"):
        """
        Save the vector representations and the FAISS index to a file for later use.

        Args:
            filepath (str, optional): The path to the file where the index should be saved. Defaults to "dpvs_index.pkl".
        """
        with open(filepath, 'wb') as f:
            #save vectors and entries together to ensure they can be reconstructed properly
            pickle.dump({'entries': self.entries, 'vectors': self.vectors}, f)

    def load(self, filepath: str="dpvs_index.pkl"):
        """
        Load the vector representations from a file and reconstruct the FAISS index.

        Args:
            filepath (str, optional): The path to the file from which the index should be loaded. Defaults to "dpvs_index.pkl".
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.entries = data['entries']
        self.vectors = data['vectors']
        self.index = self._build_faiss_index(metric=faiss.METRIC_L1, ef_construction=self.ef_construction, M=self.M, ef=self.ef)

    def lookup(self, queries: list[str], k: int=5):
        """
        Perform a similarity search on the index for a given set of queries.

        Args:
            queries (list[str]): A list of string queries to look up in the index.
            k (int, optional): The number of nearest neighbors to retrieve for each query. Defaults to 10.

        Returns:
            list[tuple[str, list[tuple[str, float]]]]: A list of tuples, where each tuple contains:
                - The original query string
                - A list of `k` nearest neighbors as tuples of (matched_string, distance)
        
        Raises:
            ValueError: If the index has not been built yet.
        """
        if self.index is None:
            raise ValueError("The index has not been built yet. Please call the `build` method before performing lookups.")
        
        query_vectors = np.array([self.vectorize(q) for q in queries], dtype=np.float32)
        distances, labels = self.index.search(query_vectors, k)
        
        results = []
        for query, idx, dists in zip(queries, labels, distances):
            result = [(self.entries[idx], dist) for idx, dist in zip(idx, dists) if idx != -1]
            results.append((query, result))
        
        return results

    def _word_vector(self, word: str):
        """
        Convert a given word into an overlapping positional, count, and neighbor-based representation float vector.

        It generates a concatenated vector with 4 distinct sub-vectors:
        1. Character frequencies
        2. Average character position
        3. Preceding characters proximity-weights
        4. Succeeding characters proximity-weights
        
        All sub-vectors are normalized by the length of the word to ensure scale invariance.

        Args:
            word (str): The string to vectorize.

        Returns:
            np.ndarray: A numpy array of type float32 representing the word.
        """
        word = word.lower()
        w_len = len(word)
        
        if w_len == 0:
            raise ValueError("The input word is empty and cannot be vectorized.")
        
        vec_cnt = np.zeros(self.chars_len, dtype=np.float32)     # Vector based on char count
        vec_pos  = np.zeros(self.chars_len, dtype=np.float32)    # Vector based on char position

        for i, ch in enumerate(word, start=1):
            if ch in self.char_idx:
                idx = self.char_idx[ch]
                vec_cnt[idx] += 1 / w_len
                vec_pos[idx]  += i / w_len
                
        # Context-based vectors
        DECAY = 0.75    # Reduces the influence of farther characters
        BOOST = 3.5     # Amplifies the influence of neighboring characters
        
        vec_pre = np.zeros(self.chars_len, dtype=np.float32)     # Vector based on preceding chars
        vec_suc = np.zeros(self.chars_len, dtype=np.float32)     # Vector based on succeeding chars
        
        for i, ch in enumerate(word, start=1):
            if ch in self.char_idx:
                idx = self.char_idx[ch]
                        
                for j in range(i - 1):
                    pre = word[j]
                    if pre in self.char_idx:
                        pre_idx = self.char_idx[pre]
                        weight = (vec_pos[pre_idx] + BOOST) * (DECAY ** (i - j))
                        vec_pre[idx] += weight / w_len

                for j in range(i + 1, len(word)):
                    suc = word[j]
                    if suc in self.char_idx:
                        suc_idx = self.char_idx[suc]
                        weight = (vec_pos[suc_idx] + BOOST) * (DECAY ** (j - i))
                        vec_suc[idx] += weight / w_len
        
        vector = np.concatenate([vec_cnt, vec_pos, vec_pre, vec_suc])
        return vector
    
    def _sentence_vector(self, sentence: str):
        """
        Convert a given sentence into a vector by averaging the vectors of its individual words.
        
        Args:
            sentence (str): The input sentence to vectorize.
            
        Returns:
            np.ndarray: A numpy array of type float32 representing the sentence.
        """
        
        words = sentence.split()
        vecs = [self._word_vector(w) for w in words if w]
        
        if not vecs: 
            raise ValueError("The input sentence does not contain any valid words for vectorization.")
        
        return np.mean(vecs, axis=0)

    def _build_entries_vectors(self, entries: list[str]):
        """
        Vectorize all the valid items in the provided entries using, and stack them into a matrix.
        
        Args:
            entries (list[str]): A list of strings to vectorize.
                                               
        Returns:
            np.ndarray: Matrix containing the vertical stack of the generated vectors.
        """
        vectors = []
        
        for w in entries:
            w = w.lower()
            vectors.append(self.vectorize(w).astype(np.float32))

        return np.vstack(vectors)
    
    def _build_faiss_index(self, metric, ef_construction, M, ef):
        """
        Construct the FAISS HNSW Index based on the built corpus vectors.
        
        Args:
            metric: The FAISS metric to use (e.g. faiss.METRIC_L1).
            ef_construction (int): The index construction depth configuration.
            M (int): The number of bi-directional links created for every new element.
            ef (int): The search depth configuration.
            
        Returns:
            faiss.Index: The constructed FAISS index.
        """
        scaled_vectors = self.vectors.copy()
            
        dim = scaled_vectors.shape[1]
        
        index = faiss.index_factory(dim, f"HNSW{M}", metric)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef
        
        index.add(scaled_vectors)
        return index