import numpy as np
import scipy as sp
import scipy.sparse as sps
from pauli import *
import copy
import matplotlib.pyplot as plt
from math import comb
import matplotlib.colors as mcolors
import openfermion as of
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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

    _SPARSE_LIMIT: int

    model_name: str
    N: int
    pauli_strings: list[str]
    H_terms: list[np.ndarray]
    H2_terms: list[np.ndarray]
    H_fulls: list[np.ndarray]
    training_grid: list[list[complex]]
    training_grid2: list[list[complex]]
    opt_basis: np.ndarray
    overlap: np.ndarray
    reduced_terms: list[np.ndarray]
    particle_selection: tuple[int, int] | int = None
    basis_ordering: str
    sparse: bool
    log: bool
    save_folder: str
    size: int

    def __init__(
        self,
        model_name: str,
        N: int,
        pauli_strings: list[str],
        training_grid: list[list[complex]],
        particle_selection: tuple[int, int] | int = None,
        basis_ordering: str = "uudd",
        log: bool = True
    ):
        self._SPARSE_LIMIT = 8

        self.model_name = model_name

        self.N = N
        if self.N > self._SPARSE_LIMIT:
            self.sparse = True
        else:
            self.sparse = False

        self.pauli_strings = pauli_strings
        self.H_terms = None
        self.H2_terms = None
        self.H_fulls = None
        self.training_grid = np.array(training_grid, dtype=complex)
        self.training_grid2 = None
        self.opt_basis = None
        self.overlap = None
        self.reduced_terms = None
        self.particle_selection = particle_selection
        self.basis_ordering = basis_ordering
        self.log = True

        if type(self.particle_selection) == type(None):
            self.size = 2**N
        elif type(self.particle_selection) == int:
            self.size = comb(N, self.particle_selection)
        elif type(self.particle_selection) == tuple:
            self.size = (
                comb(N // 2, self.particle_selection[0])
                * comb(N // 2, self.particle_selection[1])
            )
        else:
            raise Exception(
                "Particle selection should be an int, a tuple, or None"
            )

        self.save_folder = self.model_name + "_" + "N" + "_" + str(self.N)

        if self.log:
            if not os.path.isdir(self.save_folder):
                os.mkdir(self.save_folder)
            self.training_grid.tofile(
                (self.save_folder + "/" + "training_grid_"
                 + datetime.now().isoformat() + ".bin").replace(":", ".")
            )
        
        self._append_to_log(f"Model initialized with: {self.pauli_strings}")

    def _append_to_log(
        self,
        text: str
    ):
        if self.log:
            if not os.path.isdir(self.save_folder):
                os.mkdir(self.save_folder)
            with open(self.save_folder + "/surrogate.log", "a") as f:
                f.write(text + "\n")

    def build_terms(
        self,
        pregenerate_fulls: bool = False,
        save: bool = False,
        log=False
    ):
        start_time = time.perf_counter()
        self.H_terms = []
        for pauli_string in self.pauli_strings:
            if save:
                if pauli_string == "":
                    filename = self.save_folder + "/I.bin"
                else:
                    filename = self.save_folder + "/" + pauli_string + ".bin"

                try:
                    H_term = np.fromfile(filename, dtype=float).reshape(
                        (self.size, self.size)
                    )
                    self.H_terms.append(np.astype(H_term, complex))
                except:
                    self.H_terms.append(
                        gen_from_pauli_string(
                            self.N,
                            pauli_string,
                            self.particle_selection,
                            ordering=self.basis_ordering,
                            sparse=self.sparse
                        ),
                    )

                    np.astype(self.H_terms[-1], float).tofile(filename)

            else:
                self.H_terms.append(
                    gen_from_pauli_string(
                        self.N,
                        pauli_string,
                        self.particle_selection,
                        ordering=self.basis_ordering,
                        sparse=self.sparse
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

        if pregenerate_fulls:
            self.H_fulls = []
            for i in range(len(self.training_grid)):
                self.H_fulls.append(self._build_H_full(i))

        end_time = time.perf_counter()

        self._append_to_log(
            "H Terms build in: " + str(end_time - start_time) + " seconds"
        )

    def _build_H_full(
        self,
        parameter_idx: int
    ) -> np.ndarray:
        H_full = np.zeros((self.size, self.size), dtype=complex)
        for p, h in zip(self.training_grid[parameter_idx], self.H_terms):
            H_full += p * h

        return H_full

    def _calculate_residue2(
        self,
        j: int,
        basis: np.ndarray = None,
        overlap: np.ndarray = None,
        Hr_terms: list[np.ndarray] = None,
        H2r_terms: list[np.ndarray] = None,
        degeneracy_truncation: float = None
    ):
        Hr = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)
        for p, hr in zip(self.training_grid[j], Hr_terms):
            Hr += p * hr
        H2r = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)
        for p2, h2r in zip(self.training_grid2[j], H2r_terms):
            H2r += p2 * h2r

        evals, evecs = sp.linalg.eigh(Hr, overlap)

        # find degeneracy of the ground state
        degeneracy = 0
        eps = 1e-10 # for comparing floating points of GSE
        for e in evals:
            # absolute value is not needed here, e >= evals[0]
            if e - evals[0] < eps:
                degeneracy += 1
            else:
                break
            if degeneracy >= degeneracy_truncation:
                break

        # calculate residue
        res2 = 0
        for k in range(degeneracy):
            res2 += (
                evecs[:, k].conj().T
                @ (H2r - ((evals[k] * evals[k]) * overlap))
                @ evecs[:, k]
            )

        return res2


    def optimize(
        self,
        residue_threshold: float = 1e-6,
        init_vec: np.ndarray = None,
        solution_grid: tuple[np.ndarray, np.ndarray] = None,
        svd_tolerance: float = 1e-8,
        degeneracy_truncation: int = 5,
        save=False,
        processes=1
    ):
        start_time = time.perf_counter()

        if processes > 1:
            ppe = ProcessPoolExecutor(processes)
        if save:
            save_folder = self.model_name + "_" + "N" + "_" + str(self.N)
            filename = save_folder + "/basis.bin"
            try:
                flat = np.astype(
                    np.fromfile(filename, dtype=float),
                    complex
                )
                num_basis_vecs = flat.shape[0] // self.size
                self.opt_basis = flat.reshape(self.size, num_basis_vecs)
                self.overlap = self.opt_basis.conj().T @ self.opt_basis

                return self.opt_basis
            except:
                if not os.path.isdir(save_folder):
                    os.mkdir(save_folder)
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

        self._append_to_log("Initializing Optimization")
        # initial vector is not provided, so we choose from the training grid
        if type(init_vec) == type(None):
            if type(self.H_fulls) == type(None):
                H_full = self._build_H_full(0)
            else:
                H_full = self.H_fulls[0]
            if self.sparse:
                evals, evecs = sps.linalg.eigsh(H_full.real)
            else:
                evals, evecs = sp.linalg.eigh(H_full)
            init_vec = evecs[:, 0]
            chosen.append(0)
            not_chosen.remove(0)

            self._append_to_log("Chose param point 0")
        else:
            self._append_to_log("Given starting vector: " + str(init_vec))

        basis_list = [init_vec]
        basis = np.array(basis_list).T

        if type(solution_grid) != type(None):
            self._graph_solution_comparison(solution_grid, basis, chosen)

        # iteration
        num_iterations = len(not_chosen)
        for i in range(num_iterations):
            self._append_to_log(f"Iteration {i + 1}")
            overlap = (basis.conj().T @ basis).real
            max_res2 = -np.inf
            next_choice = None
            residues = []
            start_time_Hr = time.perf_counter()
            Hr_terms = []
            H2r_terms = []

            for h in self.H_terms:
                Hr_terms.append(basis.conj().T @ h @ basis)
            for h2 in self.H2_terms:
                    H2r_terms.append(basis.conj().T @ h2 @ basis)
            end_time_Hr = time.perf_counter()
            self._append_to_log(
                f"Build Hr and H2r in {end_time_Hr - start_time_Hr} seconds"
            )

            start_time_residual = time.perf_counter()
            if processes == 1:
                for j in not_chosen:
                    residues.append(self._calculate_residue2(
                        j,
                        basis,
                        overlap,
                        Hr_terms,
                        H2r_terms,
                        degeneracy_truncation
                    ))
            else:
                residues = list(ppe.map(
                    partial(
                        self._calculate_residue2,
                        basis=basis,
                        overlap=overlap,
                        Hr_terms=Hr_terms,
                        H2r_terms=H2r_terms,
                        degeneracy_truncation=degeneracy_truncation
                    ),
                    not_chosen
                ))

            max_res2 = np.max(residues)
            next_choice = not_chosen[np.argmax(residues)]

            end_time_residual = time.perf_counter()

            self._append_to_log(
                "Residual Calculation took "
                + f"{end_time_residual - start_time_residual} seconds"
            )
            self._append_to_log(f"{len(not_chosen)} residuals calculated")
            self._append_to_log(
                f"Max residual {max_res2} for param point {next_choice}"
            )
                

            print("Max Residue", max_res2)
            print("Number of residues calculated:", len(residues))

            if type(self.H_fulls) == type(None):
                chosen_H_full = self._build_H_full(next_choice)
            else:
                chosen_H_full = self.H_fulls[next_choice]

            if self.sparse:
                evals, evecs = sps.linalg.eigsh(chosen_H_full.real)
            else:
                evals, evecs = sp.linalg.eigh(chosen_H_full)

            if max_res2 < residue_threshold or len(chosen) >= 2**self.N - 1:
                end_time = time.perf_counter()

                self._append_to_log(
                    f"Optimization complete in {end_time - start_time} seconds"
                )

                print("Optimization complete.")
                plt.plot(not_chosen, np.array(residues).real, "o-")
                plt.plot(
                    [next_choice],
                    [max_res2.real],
                    "rx",
                    label="Next Choice",
                )
                plt.xlabel("Training Grid Index")
                plt.ylabel("Residue")
                plt.title(f"Termination Residues")
                plt.show()

                if type(solution_grid) != type(None):
                    self._graph_solution_comparison(
                        solution_grid, basis, chosen
                    )

                break

            plt.plot(not_chosen, np.array(residues).real, "o-")
            plt.plot([next_choice], [max_res2.real], "rx", label="Next Choice")
            plt.xlabel("Training Grid Index")
            plt.ylabel("Residue")
            plt.title(f"It {i + 1} Residues")
            plt.show()

            print("Full system size:", self.size)

            # find degeneracy of the ground state
            eps = 1e-10 # for comparing floating points of GSE
            degeneracy = 0
            for e in evals:
                if e - evals[0] < eps:
                    degeneracy += 1
                else:
                    break
                if degeneracy >= degeneracy_truncation:
                    break

            print("Degeneracy of chosen H_full ground state:", degeneracy)
            basis_addition = evecs[:, 0:degeneracy]

            # compress the basis
            projection = basis_addition - basis @ sp.linalg.solve(
                overlap, basis.conj().T @ basis_addition
            )

            U, sigmas, Vdagger = np.linalg.svd(projection)
            compress_add = 0
            for s in sigmas:
                if s > svd_tolerance:
                    compress_add += 1
                else:
                    break

            for j in range(compress_add):
                basis_list += [U[:, j]]

            basis_reduced = np.array(basis_list).T
            print("Basis size before compression:", basis.shape[1])
            print("Basis size after compression:", basis_reduced.shape[1])

            if basis_reduced.shape[1] <= basis.shape[1]:
                print(
                    "Warning: Basis did not increase in size after compression."
                )
            else:
                basis = copy.copy(basis_reduced)

            print(
                "Looking at solutions with current basis of size",
                basis.shape[1]
            )
            if type(solution_grid) != type(None):
                self._graph_solution_comparison(
                    solution_grid, basis, chosen, next_choice
                )

            not_chosen.remove(next_choice)
            chosen.append(next_choice)

        self.opt_basis = basis
        self.overlap = basis.conj().T @ basis
        self.reduced_terms = None

        if save:
            np.astype(self.opt_basis, float).tofile(filename)

        return basis

    def _graph_solution_comparison(
        self,
        solution_grid: np.ndarray,
        basis: np.ndarray,
        chosen:  list[int],
        next_choice: int = None
    ):
        answer_grid = np.zeros(
            solution_grid[0].shape,
            dtype=complex
        )
        for y in range(0, solution_grid[0].shape[0]):
            for x in range(0, solution_grid[0].shape[1]):
                if type(self.H_fulls) == type(None):
                    H_full = self._build_H_full(
                        y * solution_grid[0].shape[1] + x
                    )
                else:
                    H_full = self.H_fulls[y * solution_grid[0].shape[1] + x]

                Hr = basis.conj().T @ H_full @ basis
                overlap = basis.conj().T @ basis
                evals, evecs = sp.linalg.eigh(Hr, overlap)
                answer_grid[y, x] = evals[0]

        plt.imshow(
            np.abs((answer_grid - solution_grid[0]).real) + 1e-14,
            norm=mcolors.LogNorm(vmin=1e-14, vmax=1),
        )
        plt.colorbar(norm=mcolors.LogNorm(vmin=1e-14, vmax=1))

        if type(next_choice) != type(None):
            plt.scatter(
                next_choice % solution_grid[0].shape[1],  # type: ignore
                next_choice // solution_grid[0].shape[1],  # type: ignore
                marker="x",
                color="red",
                s=20,
                label="Next Choice",
            )

        plt.scatter(
            np.array(chosen) % solution_grid[0].shape[1],
            np.array(chosen) // solution_grid[0].shape[1],
            marker="o",
            color="orange",
            s=20,
            label="Chosen Points",
        )
        plt.xlabel(r"$\mu_2$")
        plt.ylabel(r"$\mu_1$")
        plt.title(f"Termination errors, Basis Size {basis.shape[1]}")
        plt.xticks(
            range(0, solution_grid[0].shape[1], 2),
            labels=np.round(solution_grid[1], 2)[::2],
            rotation=45,
        )
        plt.yticks(
            range(0, solution_grid[0].shape[0], 2),
            labels=np.round(solution_grid[1], 2)[::2],
            rotation=45,
        )
        plt.show()

    def solve(
        self,
        parameters: list[complex],
    ) -> complex:
        if (
            type(self.opt_basis) == type(None)
            or type(self.overlap) == type(None)
        ):
            self.optimize()

        if type(self.reduced_terms) == type(None):
            self.reduced_terms = []
            for h in self.H_terms:
                self.reduced_terms.append(
                    self.opt_basis.conj().T @ h @ self.opt_basis
                )

        Hr = np.zeros(
            (self.opt_basis.shape[1], self.opt_basis.shape[1]),
            dtype=complex
        )

        for p, h in zip(parameters, self.reduced_terms):
            Hr += p * h

        evals, evecs = sp.linalg.eigh(Hr, self.overlap)

        return evals[0]
