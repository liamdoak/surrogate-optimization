from surrogate import *
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import jordan_wigner

"""
    H_paulis = [
        "X0X1", "Y0Y1",
        "X1X2", "Y1Y2",
        "X0X2", "Y0Y2",
        "X3X4", "Y3Y4",
        "X4X5", "Y4Y5",
        "X3X5", "Y3Y5",
        "Z0Z3",
        "Z1Z4",
        "Z2Z5"
    ]
    FH = fermi_hubbard(3, 1, 1, 1)
    jw = jordan_wigner(FH)
    print(jw.terms)
"""

if __name__ == "__main__":
    ###############################################################
    # Residue threshold for terminating optimization (lower means more accurate,
    # at the cost of more basis vectors)
    res_thresh = 1e-3

    # For removing linear dependence in basis vectors
    svd_tol = 1e-8

    # Parameter grid values for training grid
    mu = np.linspace(-3.0, 3.0, 20)

    # Number of sites (total for TFIM/TFXY/Heisenberg, per spin for fermi_hubbard, AIM)
    N = 4

    # None or Between 0 and N (2*N for AIM, fermi_hubbard), N for
    # TFIM/TFXY/Heisenberg. Can be tuple for (n_up, n_down) for AIM,
    # fermi_hubbard
    ps = None

    # AIM = Single Impurity Anderson Model, fermi_hubbard, TFIM, TFXY,
    # heisenberg
    model_type = "fermi_hubbard"

    mu = np.linspace(-5, 5, 20)

    if model_type == "TFIM":
        model_parameters = {
            "J": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "TFXY":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "heisenberg":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "Jz": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "fermi_hubbard":
        U = 1.0
        model_parameters = {
            "t": 1.0,
            "mu": 1.0,
            "U": U,
            "periodic": False,
        }
    elif model_type == "AIM":
        U = 4.0
        NI = 1
        NB = N - NI
        model_parameters = {
            "NI": NI,
            "NB": NB,
            "U": U,
            "ei": [0.0] * NI,
            "vb": np.array([0.01] * ((NB) % 2) + [1.0] * (NB - (NB) % 2)),
            "eb": np.array(
                [0.0] * ((NB) % 2)
                + [1.0] * ((NB - (NB) % 2) // 2)
                + [-1.0] * ((NB - (NB) % 2) // 2)
            ),
            "mu": U / 2,
            "periodic": False,
        }

    model_paulis = model_to_paulis(N, model_type, model_parameters)
    H_paulis = [t[0] for t in model_paulis]
    H_paulis_order = {}

    for i, pauli in enumerate(H_paulis):
        H_paulis_order[pauli] = i

    ### Any training grid can be used here, this is an example of the model
    # being parameterized over two parameters
    training_grid = []
    for m1 in mu:
        for m2 in mu:
            if model_type == "TFIM":
                model_parameters["J"] = m1
                model_parameters["h"] = m2
            elif model_type == "TFXY":
                model_parameters["Jx"] = m1
                model_parameters["Jy"] = m1
                model_parameters["h"] = m2
            elif model_type == "heisenberg":
                model_parameters["Jx"] = m1
                model_parameters["Jy"] = m1
                model_parameters["Jz"] = m2
                model_parameters["h"] = 0.1
            elif model_type == "fermi_hubbard":
                model_parameters["t"] = m1
                model_parameters["mu"] = m2
                model_parameters["U"] = U
            elif model_type == "AIM":
                model_parameters["vb"] = np.array(
                    [0.01] * ((NB) % 2) + [m1] * (NB - (NB) % 2)
                )

                model_parameters["eb"] = np.array(
                    [0.0] * ((NB) % 2)
                    + [m2] * ((NB - (NB) % 2) // 2)
                    + [-m2] * ((NB - (NB) % 2) // 2)
                )
            model_paulis = model_to_paulis(N, model_type, model_parameters)
            params = np.zeros(len(H_paulis), dtype=complex)

            for t in model_paulis:
                try:
                    params[H_paulis_order[t[0]]] = t[1]
                except:
                    raise Exception("Failed to generate all terms in model")

            if model_type == "AIM" or model_type == "fermi_hubbard":
                model_paulis = list(zip(H_paulis, params))
                id_loc = np.where(
                    np.array([t[0] for t in model_paulis]) == "")[0][0]
                grid_point1 = [model_paulis[id_loc][1]]
                grid_point = grid_point1 + [
                    model_paulis[k][1]
                    for k in range(len(model_paulis)) if k != id_loc
                ]
                training_grid.append(grid_point)
            else:
                model_paulis = list(zip(H_paulis, params))
                training_grid.append([t[1] for t in model_paulis])

    if model_type == "AIM" or model_type == "fermi_hubbard":
        surrogate_N = 2 * N
        surrogate_ord = "udud"
    else:
        surrogate_N = N
        surrogate_ord = "uudd"
    model = SurrogateModel(
        model_type,
        surrogate_N,
        H_paulis,
        training_grid,
        particle_selection=ps,
        basis_ordering=surrogate_ord,
    )

    model.build_terms()
    print("Done building terms")

    # Calculate the real solutions for testing (only for 2D parameter grids)
    solution_grid = np.zeros((len(mu), len(mu)), dtype=complex)
    for i in range(len(mu)):
        for j in range(len(mu)):
            H_full = np.zeros_like(model.H_terms[0], dtype=complex)
            parameters = training_grid[i * len(mu) + j]
            for k, h in enumerate(model.H_terms):
                H_full += parameters[k] * h
            if model.sparse:
                evals, evecs = sps.linalg.eigsh(H_full.real)
            else:
                evals, evecs = np.linalg.eigh(H_full)
            solution_grid[i, j] = evals[0]

    basis = model.optimize(
        solution_grid=(
            solution_grid,
            mu,
        ),  # Comment this out if no solution grid is desired
        svd_tolerance=svd_tol,
        residue_threshold=res_thresh,
    )
    print("Basis Size", basis.shape[1])

    ### Testing the surrogate model against random parameters
    errors = []
    all_ps = []
    for i in range(200):
        H_full = np.zeros_like(model.H_terms[0], dtype=complex)

        if model_type == "TFIM":
            J = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "J": J,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "TFXY":
            Jx = 2 * np.random.randn()
            Jy = Jx
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "heisenberg":
            Jx = 2 * np.random.randn()
            Jy = Jx
            Jz = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "Jz": Jz,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "fermi_hubbard":
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "t": 2 * np.random.randn(),
                    "mu": 2 * np.random.randn(),
                    "U": U,
                },
            )
        elif model_type == "AIM":
            vb_test = np.array(
                [0.01] * ((NB) % 2) + [2 * np.random.randn()] * (NB - (NB) % 2)
            )
            eb_r = 2.0 * np.random.randn()
            eb_test = np.array(
                [0.0] * ((NB) % 2)
                + [eb_r] * ((NB - (NB) % 2) // 2)
                + [-eb_r] * ((NB - (NB) % 2) // 2)
            )
            print("vb_test", vb_test)
            print("eb_test", eb_test)
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "NI": NI,
                    "NB": NB,
                    "U": U,
                    "ei": [0.0] * NI,
                    "vb": vb_test,
                    "eb": eb_test,
                    "mu": U / 2,
                    "periodic": False,
                },
            )

        if model_type == "AIM" or model_type == "fermi_hubbard":
            id_loc = np.where(np.array([t[0] for t in model_paulis]) == "")[0][0]
            grid_point1 = [model_paulis[id_loc][1]]
            parameters = grid_point1 + [
                model_paulis[k][1] for k in range(len(model_paulis)) if k != id_loc
            ]
        else:
            parameters = [t[1] for t in model_paulis]

        for i, h in enumerate(model.H_terms):
            H_full += parameters[i] * h

        if model.sparse:
            evals, evecs = sps.linalg.eigsh(H_full.real)
        else:
            evals, evecs = np.linalg.eigh(H_full)

        # print(parameters)
        print("Real", evals[0])
        print("Approx", model.solve(parameters))
        if abs(evals[0]) < 1e-12:
            errors.append(np.abs(evals[0] - model.solve(parameters)))
        else:
            errors.append(np.abs(evals[0] - model.solve(parameters)) / np.abs(evals[0]))
        print(
            "Relative Error",
            errors[-1],
        )
        print()
        all_ps.append(parameters)
    print("Basis Size:", basis.shape[1])
    print("Full Hilbert Size:", model.H_terms[0].shape[0])
    plt.plot(errors, "o-")
    plt.xlabel("Test Case")
    plt.ylabel("Relative Error")
    plt.title("Surrogate Model Relative Errors")
    plt.yscale("log")
    plt.ylim(1e-20, 1)
    # plt.xticks(
    #     range(len(errors)),
    #     [f"({p[0]:.2f}, {p[n2b_terms + 1]:.2f})" for p in all_ps],
    #     rotation=90,
    # )
    plt.show()
