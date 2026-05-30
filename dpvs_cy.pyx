import cython
import numpy as np
cimport numpy as np
from libc.math cimport pow

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def word_vector_cython(str word, dict char_idx, int chars_len):
    cdef int w_len = len(word)
    if w_len == 0:
        raise ValueError("The input word is empty and cannot be vectorized.")

    np_cnt = np.zeros(chars_len, dtype=np.float32)
    np_pos = np.zeros(chars_len, dtype=np.float32)
    np_pre = np.zeros(chars_len, dtype=np.float32)
    np_suc = np.zeros(chars_len, dtype=np.float32)
    np_word_indices = np.zeros(w_len, dtype=np.int32)

    cdef float[:] vec_cnt = np_cnt
    cdef float[:] vec_pos = np_pos
    cdef float[:] vec_pre = np_pre
    cdef float[:] vec_suc = np_suc
    cdef int[:] word_indices = np_word_indices
    
    cdef float decay = 0.75
    cdef float boost = 3.5
    cdef float weight
    cdef int i, j, idx, pre_idx, suc_idx
    cdef int diff
    cdef float w_len_f = <float>w_len

    cdef str ch
    
    for i in range(w_len):
        ch = word[i]
        if ch in char_idx:
            word_indices[i] = char_idx[ch]
        else:
            word_indices[i] = -1

    for i in range(w_len):
        idx = word_indices[i]
        if idx != -1:
            vec_cnt[idx] += 1.0 / w_len_f
            vec_pos[idx] += <float>(i + 1) / w_len_f

    for i in range(w_len):
        idx = word_indices[i]
        if idx != -1:
            for j in range(i):
                pre_idx = word_indices[j]
                if pre_idx != -1:
                    diff = i - j
                    weight = (vec_pos[pre_idx] + boost) * pow(decay, <float>diff)
                    vec_pre[idx] += weight / w_len_f

            for j in range(i + 1, w_len):
                suc_idx = word_indices[j]
                if suc_idx != -1:
                    diff = j - i
                    weight = (vec_pos[suc_idx] + boost) * pow(decay, <float>diff)
                    vec_suc[idx] += weight / w_len_f

    return np.concatenate([np_cnt, np_pos, np_pre, np_suc])