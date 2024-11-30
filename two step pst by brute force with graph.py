import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    """
    Calculate the fidelity of the evolved state at a given time.

    Parameters:
    hamiltonian (numpy.ndarray): The Hamiltonian matrix.
    initial_state (numpy.ndarray): The initial quantum state (column vector).
    target_state (numpy.ndarray): The target quantum state (column vector).
    time (float): The total evolution time.

    Returns:
    float: Fidelity with the target state.
    numpy.ndarray: Evolved state.
    """
    # Calculate the time-evolution operator
    time_evolution_operator = expm(-1j * hamiltonian * time)

    # Evolve the initial state
    evolved_state = np.dot(time_evolution_operator, initial_state)

    # Calculate fidelity
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2
    return fidelity, evolved_state


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

    # Normalize the initial and target states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Initialize parameters
    total_time = 0
    max_fidelity = 0.999
    max_time = 10  # Maximum allowed time for evolution
    fidelity = 0
    time_step = 0.00001  # Time step for the second phase

    # Tracking fidelity for the plot
    time_values = []
    fidelity_values = []

    # Ask for the initial run time
    run_time = float(input("\nEnter the initial time duration to run the simulation: "))
    total_time += run_time
    fidelity, intermediate_state = calculate_fidelity(hamiltonian, initial_state, target_state, run_time)

    # Begin iterative evolution
    while fidelity < max_fidelity:
        # Calculate fidelity for the initial run time
        fidelity, evolved_state = calculate_fidelity(hamiltonian, intermediate_state, target_state, total_time - run_time)

        # Store fidelity and time for plotting
        time_values.append(total_time)
        fidelity_values.append(fidelity)

        # Print results
        print(f"\nFidelity at total time {total_time:.4f}: {fidelity:.7f}")

        # Check if fidelity has reached the target
        if fidelity >= max_fidelity:
            print(f"\nPerfect state transfer achieved at total time: {total_time:.4f}")
            break

        # Force user to modify the Hamiltonian
        print("\nThe fidelity has not reached the target. You must modify the Hamiltonian.")
        hamiltonian = input_matrix("Enter the new Hamiltonian matrix", size)

        # Run step-by-step evolution
        while total_time <= max_time:
            fidelity, evolved_state = calculate_fidelity(hamiltonian, intermediate_state, target_state, total_time - run_time)

            # Store fidelity and time for plotting
            time_values.append(total_time)
            fidelity_values.append(fidelity)

            # Print fidelity at each time step
            print(f"Fidelity at total time {total_time:.4f}: {fidelity:.7f}")

            if fidelity >= max_fidelity:
                print(f"\nPerfect state transfer achieved at total time: {total_time:.4f}")
                break

            total_time += time_step

        # Stop simulation if max time is reached
        if total_time > max_time:
            print("\nPerfect Quantum State Transfer was not achieved within the maximum time.")
        break

    # Final output and plotting
    plt.plot(time_values, fidelity_values, label="Fidelity vs Time")
    plt.xlabel(r"Time ($\hbar/J$)", usetex=True)
    plt.ylabel(r"Fidelity", usetex=True)
    plt.title("Fidelity of Quantum State Transfer Over Time")
    plt.legend()
    plt.grid()
    plt.savefig('fidelity_vs_time.pdf')
    plt.show()
