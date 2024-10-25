import numpy as np
from scipy.linalg import expm


def calculate_pst_time(hamiltonian, initial_state, target_state, time_step=0.0001, max_time=10):
    """
    Calculate the time for Perfect Quantum State Transfer (PST) by evolving the system numerically.

    Parameters:
    hamiltonian (numpy.ndarray): The Hamiltonian matrix.
    initial_state (numpy.ndarray): The initial quantum state (column vector).
    target_state (numpy.ndarray): The target quantum state (column vector).
    time_step (float): The time increment for evolution (default is 0.01).
    max_time (float): Maximum time to search for PST (default is 10).

    Returns:
    float: The time required for PST, or None if PST is not found within max_time.
    """

    # Check if the input dimensions are consistent
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix.")
    if initial_state.shape != (hamiltonian.shape[0], 1) or target_state.shape != (hamiltonian.shape[0], 1):
        raise ValueError("Initial and target states must have the same dimensions as the Hamiltonian.")

    # Normalize initial and target states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Print initial setup
    print("Hamiltonian:")
    print(hamiltonian)
    print("\nInitial state (normalized):")
    print(initial_state)
    print("\nTarget state (normalized):")
    print(target_state)

    # Iterate over time steps
    time = 0
    while time <= max_time:
        # Calculate the time-evolution operator
        time_evolution_operator = expm(-1j * hamiltonian * time)
        print(f"\nTime evolution operator at time {time:.4f}:")
        print(time_evolution_operator)

        # Evolve the initial state
        evolved_state = np.dot(time_evolution_operator, initial_state)
        print(f"\nEvolved state at time {time:.4f}:")
        print(evolved_state)

        # Calculate fidelity with the target state
        fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2
        print(f"Fidelity with target state at time {time:.4f}: {fidelity:.6f}")

        # Check if fidelity is close to 1 (indicating perfect transfer)
        if fidelity > 0.9999:  # Almost perfect match
            print(f"\nPerfect state transfer achieved at time: {time:.4f}")
            return time

        # Increment time
        time += time_step

    # If no PST found within max_time
    print("\nNo perfect state transfer found within the maximum time.")
    return None


def input_matrix(prompt, size):
    """
    Helper function to input a matrix row by row.
    """
    matrix = []
    print(f"{prompt} (enter row by row, space-separated):")
    for i in range(size):
        row = input(f"Row {i + 1}: ").split()
        matrix.append([float(x) for x in row])
    return np.array(matrix)


def input_vector(prompt, size):
    """
    Helper function to input a column vector.
    """
    vector = []
    print(f"{prompt} (enter elements, one per line):")
    for i in range(size):
        element = float(input(f"Element {i + 1}: "))
        vector.append([element])
    return np.array(vector)


# Main program
if __name__ == "__main__":
    size = int(input("Enter the size of the Hamiltonian matrix (e.g., 2 for 2x2 matrix): "))

    # Input the Hamiltonian
    hamiltonian = input_matrix("Enter the Hamiltonian matrix", size)

    # Input the initial state vector
    initial_state = input_vector("Enter the initial state vector", size)

    # Input the target state vector
    target_state = input_vector("Enter the target state vector", size)

    # Calculate the PST time
    pst_time = calculate_pst_time(hamiltonian, initial_state, target_state)

    if pst_time is not None:
        print(f"\nThe time for Perfect Quantum State Transfer (PST) is: {pst_time:.4f}")
    else:
        print("\nPerfect Quantum State Transfer was not achieved within the maximum time.")
