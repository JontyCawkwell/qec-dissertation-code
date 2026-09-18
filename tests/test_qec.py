import numpy as np
from sympy import Matrix, Rational

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
        np.asarray([
            Rational(1, 35),
            Rational(-8, 35),
            Rational(4, 5),
            Rational(-8, 5),
            1,
        ], dtype=float),
    )


def test_generic_code_dimensions():
    states, errors, matrix, rank, nullity, null_space = build_generic_code(2)

    assert states.shape == (3, 6)
    assert errors.shape == (3, 6)
    assert matrix.shape == (3, 3)
    assert rank == 2
    assert nullity == 1
    assert len(null_space) == 1


# Published results from:
# Y. Ouyang and R. Chao,
# "Permutation-Invariant Constant-Excitation Quantum Codes
# for Amplitude Damping", IEEE Transactions on Information Theory.
PUBLISHED_CODES = {
    1: [1, -1],
    2: [Rational(2, 5), -1, Rational(3, 5)],
    3: [
        Rational(-21, 32),
        Rational(99, 32),
        Rational(-55, 16),
        1,
    ],
    4: [
        Rational(84, 125),
        Rational(-456, 125),
        Rational(-152, 125),
        Rational(1368, 125),
        Rational(-969, 125),
        1,
    ],
    5: [
        Rational(-21505, 31104),
        Rational(135575, 31104),
        Rational(39875, 15552),
        Rational(-55825, 3888),
        Rational(-25375, 2592),
        Rational(5075, 144),
        Rational(-2639, 144),
        1,
    ],
}


def assert_proportional(actual, expected):
    """Check that two vectors represent the same null-space direction."""
    actual = Matrix(actual)
    expected = Matrix(expected)

    for i, value in enumerate(expected):
        if value != 0:
            scale = actual[i] / value
            break

    assert all(
        actual[i] == scale * expected[i]
        for i in range(len(expected))
    )


def test_published_ouyang_chao_codes():
    """Reproduce the published codes from Ouyang & Chao, Examples 1-5."""

    # Example 1 uses a 3-mode construction rather than the generic
    # construction used by build_generic_code(1).
    states = np.array([
        [3, 0, 0],
        [1, 1, 1],
    ])

    matrix = construct_a_matrix(states, kraus_error_patterns(1, 3))
    rank, nullity, null_space = analyse_a_matrix(matrix)

    assert matrix.shape == (1, 2)
    assert rank == 1
    assert nullity == 1
    assert_proportional(null_space[0], PUBLISHED_CODES[1])

    # Examples 2-5 use the generic construction.
    for t in range(2, 6):
        _, _, matrix, rank, nullity, null_space = build_generic_code(t)

        assert nullity == 1
        assert len(null_space) == 1
        assert_proportional(null_space[0], PUBLISHED_CODES[t])
