import numpy as np

from src.qec import (
    analyse_a_matrix,
    build_generic_code,
    construct_a_matrix,
    integer_partitions,
    kraus_error_patterns,
)


def test_integer_partitions():
    assert integer_partitions(4) == [
        [4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]
    ]


def test_kraus_pattern_count():
    assert len(kraus_error_patterns(8, 3)) == 40


def test_t7_dissertation_code():
    states = np.array([[64, 0], [56, 8], [48, 16], [40, 24], [32, 32]])
    matrix = construct_a_matrix(states, kraus_error_patterns(7, 2))
    rank, nullity, null_space = analyse_a_matrix(matrix)

    assert matrix.shape == (19, 5)
    assert rank == 4
    assert nullity == 1
    assert np.allclose(
        np.asarray(null_space[0], dtype=float).reshape(-1),
        [1 / 35, -8 / 35, 4 / 5, -8 / 5, 1],
    )


def test_generic_code_dimensions():
    states, errors, matrix, rank, nullity, null_space = build_generic_code(2)
    assert states.shape == (3, 6)
    assert errors.shape == (3, 6)
    assert matrix.shape == (3, 3)
    assert rank == 2
    assert nullity == 1
    assert len(null_space) == 1
