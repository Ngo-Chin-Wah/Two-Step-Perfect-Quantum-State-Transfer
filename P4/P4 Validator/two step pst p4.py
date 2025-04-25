import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Function to calculate the fidelity between an evolved initial state and a target state
def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    """
    Calculate the fidelity between the evolved initial state and the target state after a given time evolution.

    Parameters:
    hamiltonian (ndarray): The Hamiltonian matrix governing the system dynamics.
    initial_state (ndarray): The initial quantum state vector.
    target_state (ndarray): The target quantum state vector.
    time (float): Evolution time.

    Returns:
    tuple: Fidelity (float) between the evolved state and target state, and the evolved state (ndarray).
    """
    time_evolution_operator = expm(-1j * hamiltonian * time)  # Compute the time evolution operator
    evolved_state = np.dot(time_evolution_operator, initial_state)  # Apply evolution to the initial state
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2  # Calculate fidelity (overlap squared)
    return fidelity, evolved_state  # Return fidelity and evolved state


# Function to generate the Hamiltonian for a path graph P4 with given edge weights
def generate_p4_hamiltonian(weights):
    """
    Generate the Hamiltonian matrix for a 4-node path graph (P4) with specified edge weights.

    Parameters:
    weights (list or ndarray): List of three edge weights corresponding to edges (0-1), (1-2), and (2-3).

    Returns:
    ndarray: A 4x4 Hamiltonian matrix.
    """
    size = 4  # P4 graph has 4 nodes
    h = np.zeros((size, size))  # Initialize a 4x4 Hamiltonian matrix
    h[0, 1] = h[1, 0] = weights[0]  # Set weight between node 0 and 1
    h[1, 2] = h[2, 1] = weights[1]  # Set weight between node 1 and 2
    h[2, 3] = h[3, 2] = weights[2]  # Set weight between node 2 and 3
    return h  # Return the Hamiltonian matrix


if __name__ == "__main__":
    # Read initial and final edge weights from user input
    initial_weights = [float(x) for x in input("Enter initial edge weights (space-separated, 3 values): ").split()]
    final_weights = [float(x) for x in input("Enter final edge weights (space-separated, 3 values): ").split()]

    # Predefined run times for each Hamiltonian evolution phase
    run_time_1 = 9 * ((15 ** 0.5) / 4) * np.pi
    run_time_2 = 9 * ((15 ** 0.5) / 4) * np.pi
    # run_time_1 = float(input("Enter runtime before adjustment: "))
    # run_time_2 = float(input("Enter runtime after adjustment: "))

    # Generate initial and final Hamiltonians with hardcoded weights
    print(initial_weights)
    H1 = generate_p4_hamiltonian([1.0, 2 / (15) ** 0.5, 1.0])
    H2 = generate_p4_hamiltonian([-1.0, 2 / (15) ** 0.5, -1.0])

    # Define the initial state localized at node 1
    initial_state = np.array([1, 0, 0, 0])[:, np.newaxis]

    # Define the target state localized at node 4
    target_state = np.array([0, 0, 0, 1])[:, np.newaxis]

    # Normalize the initial and target states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Phase 1: Evolve the initial state under the first Hamiltonian and get the intermediate state
    _, intermediate_state = calculate_fidelity(H1, initial_state, target_state, run_time_1)

    # Phase 2: Evolve the intermediate state under the second Hamiltonian and calculate the fidelity
    fidelity_2, _ = calculate_fidelity(H2, intermediate_state, target_state, run_time_2)

    # Total fidelity after the two-step evolution
    total_fidelity = fidelity_2

    # Print the final fidelity with high precision
    print(f"\nFidelity after full runtime: {total_fidelity:.30f}")