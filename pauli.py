import numpy as np
from math import comb
from itertools import combinations
import os

class Pauli:
    I = np.array([
        [1, 0],
        [0, 1]
    ], dtype=complex)

    X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)

    Y = np.array([
        [0, -1j],
        [1j, 0]
    ])

    Z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)

def get_ps_basis(
    s: int | tuple[int],
    N: int
) -> np.ndarray:
    """
    Get the particle-number (and possibly spin) sector basis for a system of N sites.
    If the basis file does not exist, it will be calculated and saved.
    Args:
        s (int | tuple[int]): Number of particles (or tuple of spin-up and
            spin-down particles).
        N (int): Total number of sites.
    Returns:
        np.ndarray: Array of basis states in integer representation.
    """
    spin_protected = type(s) == tuple
    basis_dir = "basis_ixs"
    os.makedirs(basis_dir, exist_ok=True)
    if spin_protected:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s[0]}_{s[1]}.txt")
    else:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s}.txt")
    try:
        basis = np.loadtxt(basis_file).astype(np.int64)
        return basis

    except FileNotFoundError:
        print("No basis file, calculating now...")
        if spin_protected:
            if N % 2 != 0:
                raise ValueError("N must be even for spin-protected basis.")
            if (s[0] > N // 2) or (s[1] > N // 2):
                raise ValueError(
                    "Number of spin-up or spin-down particles exceeds half the system size."
                )

            half = N // 2
            # Generate all bit patterns for each half
            first_half = np.zeros(comb(half, s[0]), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(half), s[0])):
                n = 0
                for pos in ones_pos:
                    n |= 1 << (half - 1 - pos)
                first_half[i] = n

            second_half = np.zeros(comb(half, s[1]), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(half), s[1])):
                n = 0
                for pos in ones_pos:
                    n |= 1 << (half - 1 - pos)
                second_half[i] = n

            # Combine both halves
            basis = np.zeros(len(first_half) * len(second_half), dtype=np.int64)
            idx = 0
            for fh in first_half:
                for sh in second_half:
                    combined = (fh << half) | sh
                    basis[idx] = combined
                    idx += 1

            basis = np.sort(basis)
            np.savetxt(basis_file, basis, fmt="%d")
            return basis

        else:

            basis = np.zeros(comb(N, s), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(N), s)):  # type: ignore
                n = 0
                for pos in ones_pos:
                    n |= 1 << (N - 1 - pos)
                basis[i] = n
            basis = np.sort(basis)
            np.savetxt(basis_file, basis, fmt="%d")
            return basis

def parse_openfermion_term(
    term: tuple[int, str]
) -> str:
    string = ""
    for factor in term:
        string += factor[1] + str(factor[0])

    return pauli_strings

def gen_from_pauli_string(
    N: int,
    pauli_string: str | tuple[int, str],
    particle_selection: tuple[int] | int = None
) -> np.ndarray:
    if type(pauli_string) != str:
        pauli_string = parse_openfermion_term(pauli_string)

    next_location = 0
    mat = np.eye(1)

    for i in np.arange(0, len(pauli_string), 2):
        for j in range(int(pauli_string[i + 1]) - next_location):
            mat = np.kron(mat, Pauli.I)
        next_location = int(pauli_string[i + 1]) + 1
        if(pauli_string[i] == 'I'):
            mat = np.kron(mat, Pauli.I)
        elif(pauli_string[i] == 'X'):
            mat = np.kron(mat, Pauli.X)
        elif(pauli_string[i] == 'Y'):
            mat = np.kron(mat, Pauli.Y)
        elif(pauli_string[i] == 'Z'):
            mat = np.kron(mat, Pauli.Z)

    for i in np.arange(N - next_location):
        mat = np.kron(mat, Pauli.I)

    if type(particle_selection) != type(None):
        basis = get_ps_basis(particle_selection, N)
        mat = mat[:, basis][basis]

    return mat
