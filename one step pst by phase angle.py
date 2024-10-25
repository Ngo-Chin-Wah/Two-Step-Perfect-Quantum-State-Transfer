import numpy as np
import scipy.linalg as la

def calculate_pst_time(hamiltonian, initial_state, target_state):
    """
    Calculate the time for Perfect Quantum State Transfer (PST) given a Hamiltonian,
    initial state, and target state.

    Parameters:
    hamiltonian (numpy.ndarray): The Hamiltonian matrix.
    initial_state (numpy.ndarray): The initial quantum state (column vector).
    target_state (numpy.ndarray): The target quantum state (column vector).

    Returns:
    float: The time required for PST.
    """

    # Check if the input dimensions are consistent
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix.")
    if initial_state.shape != (hamiltonian.shape[0], 1) or target_state.shape != (hamiltonian.shape[0], 1):
        raise ValueError("Initial and target states must have the same dimensions as the Hamiltonian.")

    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = la.eigh(hamiltonian)
    print("Eigenvalues of the Hamiltonian:")
    print(eigenvalues)
    print("\nEigenvectors of the Hamiltonian (each column is an eigenvector):")
    print(eigenvectors)

    # Decompose the initial state in the eigenbasis
    initial_coeffs = np.dot(eigenvectors.T.conj(), initial_state)
    print("\nInitial state in the eigenbasis (coefficients):")
    print(initial_coeffs)

    # Decompose the target state in the eigenbasis
    target_coeffs = np.dot(eigenvectors.T.conj(), target_state)
    print("\nTarget state in the eigenbasis (coefficients):")
    print(target_coeffs)

    # Calculate phase differences for PST
    phase_diffs = np.angle(target_coeffs / initial_coeffs)
    print("\nPhase differences (angles in radians) between target and initial states in the eigenbasis:")
    print(phase_diffs)

    # Calculate the smallest time T such that all phase differences align to multiples of pi
    min_eigenvalue_diff = np.abs(np.diff(eigenvalues).min())
    times = np.pi / min_eigenvalue_diff  # Time related to phase difference
    print(f"\nMinimum eigenvalue difference: {min_eigenvalue_diff}")
    print(f"\nCalculated PST time: {times}")

    return times


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

    print(f"\nThe time for Perfect Quantum State Transfer (PST) is: {pst_time:.4f}")
