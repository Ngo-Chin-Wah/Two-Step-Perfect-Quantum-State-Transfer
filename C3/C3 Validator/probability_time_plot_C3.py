import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Function to calculate fidelity between an evolved initial state and a target state
def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    """
    Calculate the fidelity between the evolved initial state and the target state
    after a given time evolution under a Hamiltonian.

    Parameters:
    hamiltonian (ndarray): Hamiltonian matrix governing the evolution.
    initial_state (ndarray): Initial quantum state vector.
    target_state (ndarray): Target quantum state vector.
    time (float): Evolution time.

    Returns:
    tuple: Fidelity (float) between evolved and target state, and the evolved state (ndarray).
    """
    time_evolution_operator = expm(-1j * hamiltonian * time)  # Compute time evolution operator
    evolved_state = np.dot(time_evolution_operator, initial_state)  # Apply evolution to initial state
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2  # Calculate fidelity
    return fidelity, evolved_state  # Return fidelity and evolved state

# Function to generate the Hamiltonian for a C3 graph with specified edge weights
def generate_c3_hamiltonian(weights):
    """
    Generate the Hamiltonian matrix for a 3-node cycle graph (C3) with specified edge weights.

    Parameters:
    weights (list or ndarray): List of three edge weights corresponding to edges (0-1), (1-2), and (2-0).

    Returns:
    ndarray: A 3x3 Hamiltonian matrix.
    """
    size = 3  # Number of nodes
    h = np.zeros((size, size))  # Initialize 3x3 Hamiltonian
    h[0, 1] = h[1, 0] = weights[0]  # Set weight between node 0 and 1
    h[1, 2] = h[2, 1] = weights[1]  # Set weight between node 1 and 2
    h[2, 0] = h[0, 2] = weights[2]  # Set weight between node 2 and 0
    return h  # Return the Hamiltonian

if __name__ == "__main__":
    # --- Example runtimes for two-step protocol on C3 ---
    run_time_1 = 2.19420324 * 2 + 32  # Runtime for first phase
    run_time_2 = 0  # Runtime for second phase

    # Generate initial and final Hamiltonians for C3
    H1 = generate_c3_hamiltonian([1.0, 4.190286191473313, 3.0461851966572087])
    H2 = generate_c3_hamiltonian([1.0, 4.190286191473313, -3.0461851966572087])

    # Define initial state localized at node 1
    initial_state = np.array([1, 0, 0])[:, np.newaxis]

    # Define target state localized at node 3
    target_state = np.array([0, 0, 1])[:, np.newaxis]

    # Normalize initial and target states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Phase 1: Evolve initial state under first Hamiltonian
    fidelity_phase1, intermediate_state = calculate_fidelity(
        H1, initial_state, target_state, run_time_1
    )

    # Phase 2: Evolve intermediate state under second Hamiltonian
    fidelity_phase2, _ = calculate_fidelity(
        H2, intermediate_state, target_state, run_time_2
    )

    # Total fidelity after full runtime
    total_fidelity = fidelity_phase2

    # Print final fidelity with high precision
    print(f"Fidelity after full runtime: {total_fidelity:.30f}")

    # -------------------
    # PLOT PROBABILITY AT NODE 3 VS TIME
    # -------------------
    total_time = run_time_1 + run_time_2  # Total evolution time
    time_points = np.linspace(0, total_time, 50000)  # Time sampling points
    prob_node_3 = []  # Probability at node 3 over time

    # Compute probability at node 3 for each time point
    for t in time_points:
        if t <= run_time_1:
            U_t = expm(-1j * H1 * t)
            state_t = np.dot(U_t, initial_state)
        else:
            U_t = expm(-1j * H2 * (t - run_time_1))
            state_t = np.dot(U_t, intermediate_state)
        prob_node_3.append(np.abs(state_t[2])**2)

    # Probability at node 3 exactly at switching time
    switch_prob = np.abs((expm(-1j * H1 * run_time_1).dot(initial_state))[2])**2

    # Plot probability vs time
    plt.plot(time_points, prob_node_3)
    # plt.scatter(run_time_1, switch_prob, color='red', zorder=5)  # (Optional) mark switching point

    plt.xlabel('Time')
    plt.ylabel('Probability at Node 3')
    plt.title(r'$C_3$: $\alpha=4.190286191473313$, $\beta=3.0461851966572087$')
    plt.savefig("figure.pdf")  # Save figure to file
    plt.show()
