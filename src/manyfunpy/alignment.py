"""
Sequence alignment utilities.
"""
import numpy as np


def matched_index_pairs(seq1, seq2):
    """
    Return matched token indices from the longest common subsequence.
    """
    # Normalize input sequences.
    seq1 = list(seq1)
    seq2 = list(seq2)

    # Build the dynamic-programming table.
    table = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=int)
    for i in range(len(seq1) - 1, -1, -1):
        for j in range(len(seq2) - 1, -1, -1):
            if seq1[i] == seq2[j]:
                table[i, j] = table[i + 1, j + 1] + 1
            else:
                table[i, j] = max(table[i + 1, j], table[i, j + 1])

    # Trace back one stable match path.
    idx1 = []
    idx2 = []
    i = 0
    j = 0
    while i < len(seq1) and j < len(seq2):
        if seq1[i] == seq2[j]:
            idx1.append(i)
            idx2.append(j)
            i += 1
            j += 1
        elif table[i + 1, j] >= table[i, j + 1]:
            i += 1
        else:
            j += 1

    return np.asarray(idx1, dtype=int), np.asarray(idx2, dtype=int)


def align_tokens(seq1, seq2, gap=None):
    """
    Return two aligned token lists with gap placeholders.
    """
    # Find the matched token path.
    seq1 = list(seq1)
    seq2 = list(seq2)
    idx1, idx2 = matched_index_pairs(seq1, seq2)

    # Expand the matched path into two aligned lists.
    aligned1 = []
    aligned2 = []
    i = 0
    j = 0
    for k1, k2 in zip(idx1, idx2):
        while i < k1:
            aligned1.append(seq1[i])
            aligned2.append(gap)
            i += 1
        while j < k2:
            aligned1.append(gap)
            aligned2.append(seq2[j])
            j += 1
        aligned1.append(seq1[k1])
        aligned2.append(seq2[k2])
        i = k1 + 1
        j = k2 + 1
    while i < len(seq1):
        aligned1.append(seq1[i])
        aligned2.append(gap)
        i += 1
    while j < len(seq2):
        aligned1.append(gap)
        aligned2.append(seq2[j])
        j += 1

    return aligned1, aligned2


def matched_times(seq1, times1, seq2, times2):
    """
    Return matched times for two token-time sequences.
    """
    # Match sequence indices.
    idx1, idx2 = matched_index_pairs(seq1, seq2)

    # Select the matched timestamps.
    times1 = np.asarray(times1, dtype=float)
    times2 = np.asarray(times2, dtype=float)

    return times1[idx1], times2[idx2], idx1, idx2
