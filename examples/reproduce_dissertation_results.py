"""Reproduce the novel t=6, 7 and 8 codes reported in the dissertation."""

import numpy as np

from src.qec import analyse_a_matrix, construct_a_matrix, kraus_error_patterns


CODES = {
    6: np.array([
        [49, 0, 0], [42, 7, 0], [35, 14, 0], [35, 7, 7],
        [28, 21, 0], [28, 14, 7], [21, 21, 7],
    ]),
    7: np.array([
        [64, 0], [56, 8], [48, 16], [40, 24], [32, 32],
    ]),
    8: np.array([
        [81, 0, 0], [72, 9, 0], [63, 18, 0], [63, 9, 9],
        [54, 27, 0], [54, 18, 9], [45, 36, 0], [45, 27, 9],
        [36, 36, 9],
    ]),
}


EXPECTED_NULL_SPACES = {
    6: [1 / 20, -7 / 20, 9 / 20, 3 / 5, -1 / 4, -3 / 2, 1],
    7: [1 / 35, -8 / 35, 4 / 5, -8 / 5, 1],
    8: [-1 / 70, 9 / 70, -2 / 7, -8 / 35, 2 / 5, 4 / 5, -1 / 5, -8 / 5, 1],
}


for t, states in CODES.items():
    error_patterns = kraus_error_patterns(t, states.shape[1])
    a_matrix = construct_a_matrix(states, error_patterns)
    rank, nullity, null_space = analyse_a_matrix(a_matrix)

    print(f"t={t}: shape={a_matrix.shape}, rank={rank}, nullity={nullity}")
    print(f"null space = {null_space}")

    calculated = np.asarray(null_space[0], dtype=float).reshape(-1)
    expected = np.asarray(EXPECTED_NULL_SPACES[t], dtype=float)
    assert np.allclose(calculated, expected)

print("All dissertation results reproduced successfully.")
