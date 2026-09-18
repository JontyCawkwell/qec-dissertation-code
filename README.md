# Quantum Error Correction: Extending Codes to Six, Seven, and Eight Errors

This project implements the method from my physics dissertation, extending a family of quantum error correction codes to correct six, seven, and eight errors, beyond the previously published maximum of five.

## The problem

Quantum information encoded in these codes is vulnerable to two different kinds of error. Amplitude-damping errors arise when photons are lost during storage, causing the system to decay toward its ground state. Permutation errors arise during transmission, when the roles of individual modes get mixed up. This project focuses on protecting against amplitude-damping errors. The codes are built from a symmetric, "permutation-invariant" family, so they are inherently robust to permutation errors as well, while keeping the amplitude-damping problem computationally tractable even as the number of correctable errors grows.

Whether a valid code exists for a given number of correctable errors comes down to solving a matrix problem, and I built the code to construct and solve that matrix computationally. Finding new codes for six, seven, and eight errors meant solving this at a larger scale than had previously been demonstrated for this code family.

## What's in this repo

- `src/qec.py`: core functions for building candidate states, generating error patterns, constructing the A-matrix, and analysing its rank and null space.
- `tests/test_qec.py`: unit tests, including reproduction of the five previously published codes (Ouyang & Chao) as a correctness check on the method, plus a check that the `t = 7` novel code matches the exact result reported in the dissertation.
- `examples/reproduce_dissertation_results.py`: a script that reconstructs the `t = 6, 7, 8` codes and verifies each against the dissertation's reported results.
- `notebooks/reproduce_results.ipynb`: the same walkthrough in an interactive notebook, with explanation alongside the code.

## How it works

For a given number of correctable errors `t`, the code constructs an **A-matrix** from the candidate quantum states and the possible error patterns. A valid code exists if this matrix has a non-trivial null space, and the vectors in that null space are the actual coefficients of a valid error-correcting code.

## Validation

Before using this method to find anything new, the test suite first reproduces the five codes already published in Ouyang & Chao, *"Permutation-Invariant Constant-Excitation Quantum Codes for Amplitude Damping"* (IEEE Transactions on Information Theory). Only once those known results matched exactly did I apply the same method to `t = 6, 7, 8`, which is where the dissertation's novel contribution begins.

## Running it

```bash
pip install -r requirements.txt

# Reproduce and verify the three novel results
python -m examples.reproduce_dissertation_results

# Run the test suite, including the published-code validation
pytest
```

## Background

This code supports the quantitative claims in my undergraduate dissertation in Physics at the University of Sheffield (First Class). The method follows the Knill-Laflamme framework for quantum error correction, applied to the permutation-invariant, constant-excitation code family introduced by Y. Ouyang and R. Chao.
