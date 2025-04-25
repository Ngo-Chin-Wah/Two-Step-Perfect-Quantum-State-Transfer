import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Function to calculate fidelity between an evolved initial state and a target state
def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    """
    Calculate the fidelity between the evolved initial state and the target state
    after a given time evolution under a Hamiltonian.

    Parameters:
    hamiltonian (ndarray): The Hamiltonian matrix governing system evolution.
    initial_state (ndarray): The initial quantum state vector.
    target_state (ndarray): The target quantum state vector.
    time (float): Evolution time.

    Returns:
    tuple: Fidelity (float) between evolved and target state, and the evolved state (ndarray).
    """
    time_evolution_operator = expm(-1j * hamiltonian * time)  # Compute time evolution operator
    evolved_state = np.dot(time_evolution_operator, initial_state)  # Apply evolution to initial state
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2  # Compute fidelity (overlap squared)
    return fidelity, evolved_state  # Return fidelity and evolved state

# Function to generate the Hamiltonian for a modified P4-like graph (3 nodes connected pairwise)
def generate_c3_hamiltonian(weights):
    """
    Generate the Hamiltonian matrix for a 3-node graph with specified edge weights.

    Parameters:
    weights (list or ndarray): List of three edge weights corresponding to edges (0-1), (1-2), and (0-2).

    Returns:
    ndarray: A 3x3 Hamiltonian matrix.
    """
    size = 3  # Number of nodes
    h = np.zeros((size, size))  # Initialize 3x3 Hamiltonian
    h[0, 1] = h[1, 0] = weights[0]  # Set weight between node 0 and node 1
    h[1, 2] = h[2, 1] = weights[1]  # Set weight between node 1 and node 2
    h[0, 2] = h[2, 0] = weights[2]  # Set weight between node 0 and node 2
    return h  # Return the Hamiltonian

if __name__ == "__main__":
    # Predefined run times for each Hamiltonian evolution phase
    run_time_1 = 1.047188304150996
    run_time_2 = 1.047188304150996

    # Generate initial and final Hamiltonians
    H1 = generate_p4_hamiltonian([1.0, 2.0916222079639333, 2.0917650639286895])
    H2 = generate_p4_hamiltonian([1.0, 2.0916222079639333, -2.0917650639286895])

    # Define initial state localized at node 1
    initial_state = np.array([1, 0, 0])[:, np.newaxis]

    # Define target state localized at node 2
    target_state = np.array([0, 1, 0])[:, np.newaxis]

    # Normalize initial and target states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Phase 1: Evolve the initial state under the first Hamiltonian to get intermediate state
    _, intermediate_state = calculate_fidelity(H1, initial_state, target_state, run_time_1)

    # Phase 2: Evolve intermediate state under the second Hamiltonian and calculate final fidelity
    fidelity_2, _ = calculate_fidelity(H2, intermediate_state, target_state, run_time_2)

    # Total fidelity after two-step evolution
    total_fidelity = fidelity_2

    # Print final fidelity with high precision
    print(f"\nFidelity after full runtime: {total_fidelity:.30f}")
