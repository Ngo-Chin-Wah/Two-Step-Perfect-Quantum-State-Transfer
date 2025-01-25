import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    time_evolution_operator = expm(-1j * hamiltonian * time)
    evolved_state = np.dot(time_evolution_operator, initial_state)
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2
    return fidelity, evolved_state


def generate_p4_hamiltonian(weights):
    size = 4
    h = np.zeros((size, size))
    h[0, 1] = h[1, 0] = weights[0]
    h[1, 2] = h[2, 1] = weights[1]
    h[2, 3] = h[3, 2] = weights[2]
    return h


if __name__ == "__main__":
    initial_weights = [float(x) for x in input("Enter initial edge weights (space-separated, 3 values): ").split()]
    final_weights = [float(x) for x in input("Enter final edge weights (space-separated, 3 values): ").split()]
    run_time_1 = float(input("Enter runtime before adjustment: "))
    run_time_2 = float(input("Enter runtime after adjustment: "))

    # Initial and final Hamiltonians
    H1 = generate_p4_hamiltonian(initial_weights)
    H2 = generate_p4_hamiltonian(final_weights)

    # Initial state (excitation at node 1)
    initial_state = np.array([1, 0, 0, 0])[:, np.newaxis]

    # Target state (excitation at node 4)
    target_state = np.array([0, 0, 0, 1])[:, np.newaxis]

    # Normalize states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Fidelity and intermediate state calculation (phase 1)
    _, intermediate_state = calculate_fidelity(H1, initial_state, target_state, run_time_1)

    # Fidelity calculation (phase 2) using the intermediate state
    fidelity_2, _ = calculate_fidelity(H2, intermediate_state, target_state, run_time_2)

    # Total fidelity
    total_fidelity = fidelity_2

    print(f"\nFidelity after full runtime: {total_fidelity:.30f}")
