"""Utilities for constructing permutation-invariant, constant-excitation QEC codes.

The functions in this module implement the A-matrix method used in the dissertation
"Permutation-Invariant Constant-Excitation Quantum Codes for Amplitude Damping".
"""

from fractions import Fraction
from math import comb

import numpy as np
from sympy import Matrix
from sympy.utilities.iterables import multiset_permutations


def integer_partitions(n: int, max_value: int | None = None) -> list[list[int]]:
    """Return the integer partitions of ``n`` in non-increasing order."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if max_value is None:
        max_value = n
    if n == 0:
        return [[]]

    partitions = []
    for value in range(min(n, max_value), 0, -1):
        for remainder in integer_partitions(n - value, value):
            partitions.append([value] + remainder)
    return partitions


def kraus_error_patterns(t: int, m: int) -> np.ndarray:
    """Generate Kraus error patterns for up to ``t`` amplitude-damping errors.

    Each row represents an integer partition of an error weight, padded with
    zeros to ``m`` modes. Partitions requiring more than ``m`` modes are omitted.
    The zero-error case is excluded, as in the dissertation.
    """
    if t < 1 or m < 1:
        raise ValueError("t and m must be positive")

    patterns = [
        partition
        for error_weight in range(1, t + 1)
        for partition in integer_partitions(error_weight)
        if len(partition) <= m
    ]

    result = np.zeros((len(patterns), m), dtype=int)
    for i, partition in enumerate(patterns):
        result[i, : len(partition)] = partition
    return result


def constant_excitation_states(t: int, m: int, u: int) -> np.ndarray:
    """Generate the generic constant-excitation basis states.

    The first rows are the integer partitions of ``t``, multiplied by ``u``
    and padded to ``m`` modes. The final row is the all-ones state used in the
    generic construction from the dissertation.
    """
    if t < 1 or m < 1 or u < 1:
        raise ValueError("t, m and u must be positive")

    partitions = integer_partitions(t)
    if any(len(partition) > m for partition in partitions):
        raise ValueError("m is too small for the generic basis construction")

    states = np.zeros((len(partitions) + 1, m), dtype=int)
    for i, partition in enumerate(partitions):
        states[i, : len(partition)] = np.asarray(partition) * u
    states[-1] = 1
    return states


def unique_permutations(state: np.ndarray | list[int]) -> np.ndarray:
    """Return distinct permutations of the non-zero entries of a state.

    Zeros are appended after permutation. This matches the optimisation used
    in the dissertation's A-matrix calculation because the Kraus patterns are
    represented with non-zero entries first.
    """
    nonzero = [value for value in state if value != 0]
    zero_count = len(state) - len(nonzero)
    permutations = list(multiset_permutations(nonzero))
    return np.asarray([list(p) + [0] * zero_count for p in permutations], dtype=int)


def normalisation_factor(state: np.ndarray, error_pattern: np.ndarray) -> Fraction:
    """Calculate the permutation-invariance normalisation factor."""
    permutations = list(multiset_permutations(state))
    valid = sum(
        all(c >= a for c, a in zip(permutation, error_pattern))
        for permutation in permutations
    )
    return Fraction(valid, len(permutations))


def combinatorial_factor(state: np.ndarray, error_pattern: np.ndarray) -> Fraction:
    """Calculate the averaged binomial coefficient in an A-matrix element."""
    permutations = unique_permutations(state)
    total = sum(
        _binomial_product(permutation, error_pattern) for permutation in permutations
    )
    return Fraction(total, len(permutations))


def _binomial_product(state: np.ndarray | list[int], error_pattern: np.ndarray) -> int:
    """Calculate the product of binomial coefficients for one permutation."""
    product = 1
    for excitations, errors in zip(state, error_pattern):
        if excitations < errors:
            return 0
        product *= comb(int(excitations), int(errors))
    return product


def construct_a_matrix(
    states: np.ndarray,
    error_patterns: np.ndarray,
) -> Matrix:
    """Construct the exact SymPy A-matrix for a set of basis states."""
    states = np.asarray(states, dtype=int)
    error_patterns = np.asarray(error_patterns, dtype=int)

    if states.ndim != 2 or error_patterns.ndim != 2:
        raise ValueError("states and error_patterns must be two-dimensional")
    if states.shape[1] != error_patterns.shape[1]:
        raise ValueError("states and error_patterns must have the same number of modes")

    matrix = []
    for error_pattern in error_patterns:
        row = []
        for state in states:
            value = normalisation_factor(state, error_pattern)
            value *= combinatorial_factor(state, error_pattern)
            row.append(value)
        matrix.append(row)
    return Matrix(matrix)


def analyse_a_matrix(a_matrix: Matrix) -> tuple[int, int, list[Matrix]]:
    """Return the rank, nullity and null-space basis of an A-matrix."""
    rank = a_matrix.rank()
    nullity = a_matrix.cols - rank
    return rank, nullity, a_matrix.nullspace()


def build_generic_code(t: int) -> tuple[np.ndarray, np.ndarray, Matrix, int, int, list[Matrix]]:
    """Construct and analyse the generic code for ``t`` correctable errors."""
    u = t + 1
    m = t * u
    states = constant_excitation_states(t, m, u)
    error_patterns = kraus_error_patterns(t, m)
    a_matrix = construct_a_matrix(states, error_patterns)
    rank, nullity, null_space = analyse_a_matrix(a_matrix)
    return states, error_patterns, a_matrix, rank, nullity, null_space
