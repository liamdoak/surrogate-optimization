# Bring up to Norman
# Particle Selection?
#  - get_ps_basis
#  - get_occupied_indices
#  - H[:,basis][basis[
# any good hamiltonians to test with?
##### allow "fixing" some parameters?
# multithreading (possibly with Python 3.14?)
# JIT (possibly with Python 3.14?)

import numpy as np
import scipy as sp
from pauli import *
import copy
import matplotlib.pyplot as plt

class SurrogateModel:
    """
    A class to do surrogate optimizations on a Hamiltonian and a training grid
    of parameters

    Attributes:
        N : `int`
            The number of particles in the system
        pauli_strings: `list[str]`
            A list of Pauli strings that comprise the Hamiltonian
    """
    N: int
    pauli_strings: list[str]
    H_terms: list[np.ndarray]
    H2_terms: list[np.ndarray]
    training_grid: list[list[complex]]
    training_grid2: list[list[complex]]
    opt_basis: np.ndarray
    overlap: np.ndarray
    reduced_terms: list[np.ndarray]
    particle_selection: tuple[int, int] | int = None

    def __init__(
        self,
        N: int,
        pauli_strings: list[str],
        training_grid: list[list[complex]],
        particle_selection: tuple[int, int] | int = None
    ):
        self.N = N
        self.pauli_strings = pauli_strings
        self.H_terms = None
        self.H2_terms = None
        self.training_grid = training_grid
        self.training_grid2 = None
        self.opt_basis = None
        self.overlap = None
        self.reduced_terms = None
        self.particle_selection = particle_selection

    def build_terms(
        self
    ):
        self.H_terms = []
        for pauli_string in self.pauli_strings:
            self.H_terms.append(
                gen_from_pauli_string(
                    self.N, pauli_string, self.particle_selection
                ),
            )

        self.H2_terms = []
        for h_i in self.H_terms:
            for h_j in self.H_terms:
                self.H2_terms.append(h_i @ h_j)

        self.training_grid2 = []
        for mu in self.training_grid:
            bulk = []
            for mu_i in mu:
                for mu_j in mu:
                    bulk.append(mu_i * mu_j)
            self.training_grid2.append(bulk)

    def _build_H_full(
        self,
        parameter_idx: int
    ) -> np.ndarray:
        H_full = np.zeros((2**self.N, 2**self.N), dtype=complex)
        for p, h in zip(self.training_grid[parameter_idx], self.H_terms):
            H_full += p * h

        return H_full

    def _build_H2_full(
        self,
        parameter_idx: int
    ) -> np.ndarray:
        H2_full = np.zeros((2**self.N, 2**self.N), dtype=complex)
        for mu2_i, h2_i in zip(
            self.training_grid2[parameter_idx],
            self.H2_terms
        ):
            H2_full += mu2_i * h2_i

        return H2_full

    def optimize(
        self,
        residue_threshold: int = 1,
        init_vec: np.ndarray = None,
        solution_grid: np.ndarray = None
    ):
        # build terms if they are not already built
        if (
            type(self.H_terms) == type(None)
            or type(self.H2_terms) == type(None)
            or type(self.training_grid2) == type(None)
        ):
            self.build_terms()

        # list of indices into the training grid
        chosen = []

        # list of remaining indices into the training grid
        not_chosen = list(range(len(self.training_grid)))

        # list of ill-conditioned choices
        dont_choose = []

        # initial vector is not provided, so we choose from the training grid
        if init_vec == None:
            H_full = self._build_H_full(0)
            evals, evecs = sp.linalg.eigh(H_full)
            init_vec = evecs[:, 0]
            chosen.append(0)
            not_chosen.remove(0)

        basis_list = [init_vec]
        basis = np.array(basis_list).T

        if type(solution_grid) != type(None):
            answer_grid = np.zeros(solution_grid.shape, dtype=complex)
            for y in range(solution_grid.shape[0]):
                for x in range(solution_grid.shape[1]):
                    H_full = self._build_H_full(y * 10 + x)
                    Hr = basis.conj().T @ H_full @ basis
                    overlap = basis.conj().T @ basis
                    evals, evecs = sp.linalg.eigh(Hr, overlap)
                    answer_grid[y, x] = evals[0]

            plt.imshow(np.abs((answer_grid - solution_grid).real))
            plt.xlabel("Bz")
            plt.ylabel("J")
            plt.title("It 0")
            plt.colorbar()
            plt.show()

        # iteration
        num_iterations = len(not_chosen)
        for i in range(num_iterations):
            overlap = basis.conj().T @ basis
            max_res2 = -np.inf
            next_choice = None
            chosen_H_full = None
            for j in not_chosen:
                # construct Hr and H2r
                # technically H_full and H2_full only need to be constructed
                # once, however, due to possible memory limitations based on
                # the trianing grid size and full Hilbert space size, these are
                # constructed on demand
                H_full = self._build_H_full(j)
                H2_full = self._build_H2_full(j)

                Hr = basis.conj().T @ H_full @ basis
                H2r = basis.conj().T @ H2_full @ basis
                evals, evecs = sp.linalg.eigh(Hr, overlap)

                # find degeneracy of the ground state
                eps = 1e-7 # for comparing floating points of GSE
                degeneracy = 0
                for e in evals:
                    if e - evals[0] < eps:
                        degeneracy += 1
                    else:
                        break

                # calculate residue
                res2 = 0
                for k in range(degeneracy):
                    res2 += (
                        evecs[:, k].conj().T
                        @ (H2r - evals[k] * evals[k] * overlap)
                        @ evecs[:, k]
                    )

                if res2 > max_res2:
                    max_res2 = res2
                    next_choice = j
                    chosen_H_full = H_full

            print("Max Residue", max_res2)
            evals, evecs = np.linalg.eigh(chosen_H_full)
            # find degeneracy of the ground state
            eps = 1e-7 # for comparing floating points of GSE
            degeneracy = 0
            for e in evals:
                if e - evals[0] < eps:
                    degeneracy += 1
                else:
                    break

            basis_addition = evecs[:, 0:degeneracy]

            # compress the basis
            projection = basis_addition - basis @ sp.linalg.solve(
                overlap, basis.conj().T @ basis_addition
            )

            U, sigmas, Vdagger = np.linalg.svd(projection)
            compress_add = 0
            for s in sigmas:
                if(s > eps):
                    compress_add += 1
                else:
                    break

            basis_list = copy.deepcopy(basis_list)
            for j in range(compress_add):
                basis_list += [U[:, j]]
            basis = np.array(basis_list).T

            if type(solution_grid) != type(None):
                answer_grid = np.zeros(solution_grid.shape, dtype=complex)
                for y in range(solution_grid.shape[0]):
                    for x in range(solution_grid.shape[1]):
                        H_full = self._build_H_full(y * 10 + x)
                        Hr = basis.conj().T @ H_full @ basis
                        overlap = basis.conj().T @ basis
                        evals, evecs = sp.linalg.eigh(Hr, overlap)
                        answer_grid[y, x] = evals[0]

                plt.imshow(np.abs((answer_grid - solution_grid).real))
                plt.xlabel("Bz")
                plt.ylabel("J")
                plt.title(f"It {i + 1}")
                plt.colorbar()
                plt.show()

            not_chosen.remove(next_choice)
            chosen.append(next_choice)

            if max_res2 < residue_threshold or len(chosen) >= 2**self.N - 1:
                break

        self.opt_basis = basis
        self.overlap = basis.conj().T @ basis
        self.reduced_terms = None

        return chosen, basis
    
    def solve(
        self,
        parameters: list[complex]
    ) -> complex:
        if (
            type(self.opt_basis) == type(None)
            or type(self.overlap) == type(None)
        ):
            self.optimize()

        if self.reduced_terms == None:
            self.reduced_terms = []
            for h in self.H_terms:
                self.reduced_terms.append(
                    self.opt_basis.conj().T @ h @ self.opt_basis
                )

        Hr = np.zeros(
            (self.opt_basis.shape[1], self.opt_basis.shape[1]),
            dtype = complex
        )

        for p, h in zip(parameters, self.reduced_terms):
            Hr += p * h
        
        evals, evecs = sp.linalg.eigh(Hr, self.overlap)
        
        return evals[0]

if __name__ == "__main__":
    mu = np.linspace(-3, 3, 10)

    training_grid = np.array([
        (i, i, i, i, i, i, j, j, j, j, j, j)
        for i in mu
        for j in mu
    ])
    
    H_paulis = [
        "X0X1", "X1X2", "X2X3", "X3X4", "X4X5", "X0X5",
        "Z0", "Z1", "Z2", "Z3", "Z4", "Z5"
    ]
    model = SurrogateModel(
        6,
        H_paulis,
        training_grid,
        particle_selection=3
    )

    model.build_terms()

    # calc real solutions
    solution_grid = np.zeros((10, 10), dtype=complex)
    for i in range(10):
        for j in range(10):
            H_full = np.zeros((20, 20), dtype=complex)
            parameters = training_grid[i * 10 + j]
            for k, h in enumerate(model.H_terms):
                H_full += parameters[k] * h
            evals, evecs = np.linalg.eigh(H_full)
            solution_grid[i, j] = evals[0]

    print(model.optimize(solution_grid=None))

    for i in range(30):
        H_full = np.zeros((2**6, 2**6), dtype=complex)

        (J, Bz) = 3 * np.random.randn(2)
        parameters = [J, J, J, J, J, J, Bz, Bz, Bz, Bz, Bz, Bz]
        for i, h in enumerate(model.H_terms):
            H_full += parameters[i] * h

        evals, evecs = np.linalg.eigh(H_full)

        print(parameters)
        print("Real", evals[0])
        print("Approx", model.solve(parameters))
        print("Diff", np.abs(evals[0] - model.solve(parameters)))
        print()
