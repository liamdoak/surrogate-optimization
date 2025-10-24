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
from math import comb
import matplotlib.colors as mcolors

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
    size: int

    def __init__(
        self,
        N: int,
        pauli_strings: list[str] | tuple[tuple[int, str]],
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
        if type(self.particle_selection) == type(None):
            self.size = 2**N
        elif type(self.particle_selection) == int:
            self.size = comb(N, self.particle_selection)

    def build_terms(
        self
    ):
        self.H_terms = []
        for pauli_string in self.pauli_strings:
            H = gen_from_pauli_string(
                self.N, pauli_string, self.particle_selection
            )
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
        print(self.training_grid2)

    def _build_H_full(
        self,
        parameter_idx: int
    ) -> np.ndarray:
        H_full = np.zeros((self.size, self.size), dtype=complex)
        for p, h in zip(self.training_grid[parameter_idx], self.H_terms):
            H_full += p * h

        return H_full

    def _build_H2_full(
        self,
        parameter_idx: int
    ) -> np.ndarray:
        H2_full = np.zeros((self.size, self.size), dtype=complex)
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
        solution_grid: np.ndarray = None,
        log_file = None
    ):
        if log_file:
            log_file = open(log_file, "w")
            log_file.write("# Parameter Set\n")
            for mu in self.training_grid:
                log_file.write(str(mu)+"\n")
            
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
        print(init_vec)

        if type(solution_grid) != type(None):
            answer_grid = np.zeros(solution_grid.shape, dtype=complex)
            for y in range(solution_grid.shape[0]):
                for x in range(solution_grid.shape[1]):
                    H_full = self._build_H_full(y * solution_grid.shape[1] + x)
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
                # the trianing grid size and full Hilbert space size, these
                # are constructed on demand
                H_full = self._build_H_full(j)
                H2_full = self._build_H2_full(j)

                Hr = basis.conj().T @ H_full @ basis
                H2r = basis.conj().T @ H2_full @ basis
                evals, evecs = sp.linalg.eigh(Hr, overlap)

                # find degeneracy of the ground state
                eps = 1e-8 # for comparing floating points of GSE
                degeneracy = 0
                for e in evals:
                    if e - evals[0] < eps:
                        degeneracy += 1
                    else:
                        break

                # calculate residue
                res2 = 0
                for k in range(degeneracy):
                    print(evals)
                    res2 += (
                        evecs[:, k].conj().T
                        @ (H2r - evals[k] * evals[k] * overlap)
                        @ evecs[:, k]
                    )
                print("Residue:", res2)
                if(res2 < -1e-7):
                    print("Fail")

                if res2 > max_res2:
                    max_res2 = res2
                    next_choice = j
                    chosen_H_full = H_full

            print("Max Residue", max_res2)
            evals, evecs = np.linalg.eigh(chosen_H_full)
            # find degeneracy of the ground state
            eps = 1e-8 # for comparing floating points of GSE
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

            for j in range(compress_add):
                basis_list += [U[:, j]]
            basis = np.array(basis_list).T

            if type(solution_grid) != type(None):
                answer_grid = np.zeros(solution_grid.shape, dtype=complex)
                for y in range(solution_grid.shape[0]):
                    for x in range(solution_grid.shape[1]):
                        H_full = self._build_H_full(
                            y * solution_grid.shape[1] + x
                        )
                        Hr = basis.conj().T @ H_full @ basis
                        overlap = basis.conj().T @ basis
                        evals, evecs = sp.linalg.eigh(Hr, overlap)
                        answer_grid[y, x] = evals[0]
                plt.imshow(
                    np.abs((answer_grid - solution_grid).real) + 1e-20,
                    norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                )
                plt.colorbar(
                    norm=mcolors.LogNorm(
                        vmin=1e-20, vmax=1
                    )  # , ticks=[1e-20, 1e-10, 1e0]
                )
                plt.annotate("X", xy=(next_choice % 20, next_choice // 20))
                plt.xlabel("U")
                plt.ylabel("t")
                plt.title(f"It {i + 1}")
                plt.show()

            not_chosen.remove(next_choice)
            chosen.append(next_choice)
            print(chosen)

            if max_res2 < residue_threshold or len(chosen) >= self.size - 1:
                break

        self.opt_basis = basis
        self.overlap = basis.conj().T @ basis
        plt.imshow(self.overlap.real)
        plt.colorbar()
        plt.show()
        self.reduced_terms = None

        if log_file:
            log_file.close()

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
        
        return evals[0]#, self.opt_basis @ evecs[:, 0]
