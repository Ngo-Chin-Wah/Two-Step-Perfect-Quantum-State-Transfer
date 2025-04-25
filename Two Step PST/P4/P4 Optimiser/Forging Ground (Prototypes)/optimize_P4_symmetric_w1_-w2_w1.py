import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

# Function to optimize symmetric P4 graph Hamiltonians for quantum state transfer
def fidelity_simulation_P4_symmetric_weights(fixed_initial_edge_weights, initial_runtime_before,
                                             initial_runtime_after, target_fidelity=1.0,
                                             stagnation_threshold=0.1, stagnation_window=1500):
    """
    Optimize symmetric edge weights and runtimes of a P4 graph Hamiltonian to achieve a target fidelity
    for transferring quantum state from node 1 to node 4.

    Parameters:
    fixed_initial_edge_weights (list of float): Initial guess for edge weights [w1, w2, w3] where w3 = w1.
    initial_runtime_before (float): Initial evolution time before switching.
    initial_runtime_after (float): Initial evolution time after switching.
    target_fidelity (float, optional): Target fidelity to achieve (default 1.0).
    stagnation_threshold (float, optional): Threshold for detecting stagnation (default 0.1).
    stagnation_window (int, optional): Number of iterations for stagnation check (default 1500).

    Returns:
    None: Saves solutions to CSV and plots loss history.
    """

    def build_hamiltonian_tf(edge_weights):
        """
        Build the symmetric Hamiltonian matrix for a P4 graph using TensorFlow.

        Parameters:
        edge_weights (list or Tensor): [w1, w2, w3] edge weights.

        Returns:
        tf.Tensor: 4x4 complex Hamiltonian.
        """
        edge_weights = tf.convert_to_tensor(edge_weights, dtype=tf.float64)
        h = tf.stack([[0, edge_weights[0], 0, 0],
                      [edge_weights[0], 0, edge_weights[1], 0],
                      [0, edge_weights[1], 0, edge_weights[2]],
                      [0, 0, edge_weights[2], 0]])
        return tf.cast(h, tf.complex128)

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """
        Calculate the fidelity between evolved and target states.

        Parameters:
        hamiltonian (tf.Tensor): Hamiltonian matrix.
        initial_state (tf.Tensor): Initial quantum state.
        target_state (tf.Tensor): Target quantum state.
        runtime (tf.Tensor): Evolution time.

        Returns:
        tf.Tensor: Fidelity value.
        """
        runtime = tf.cast(runtime, tf.complex128)
        time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

    # Define initial and target states
    initial_state = np.array([1, 0, 0, 0], dtype=np.complex128)
    target_state = np.array([0, 0, 1, 0], dtype=np.complex128)

    # Normalize states
    initial_state_tf = tf.constant(initial_state / np.linalg.norm(initial_state), dtype=tf.complex128)
    target_state_tf = tf.constant(target_state / np.linalg.norm(target_state), dtype=tf.complex128)

    # Define variables
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float64)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float64)
    initial_edge_weights_tf = tf.Variable(fixed_initial_edge_weights[:2], dtype=tf.float64)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Track loss, fidelity, and found solutions
    loss_history, fidelity_history, solutions = [], [], []
    stagnation_counter = 0

    # Optimization loop
    for step in range(500000):
        with tf.GradientTape() as tape:
            w1, w2 = initial_edge_weights_tf[0], initial_edge_weights_tf[1]

            # Initial symmetric weights: [w1, w2, w1]
            initial_weights = [w1, w2, w1]

            # Final symmetric weights: [w1, -w2, w1]
            final_weights = [w1, -w2, w1]

            # Build Hamiltonians
            h_initial = build_hamiltonian_tf(initial_weights)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex128)), initial_state_tf)

            h_final = build_hamiltonian_tf(final_weights)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_after)

            # Loss function: 1 - fidelity
            loss_value = 1.0 - fidelity_value

        # Apply gradients
        gradients = tape.gradient(loss_value, [runtime_before, runtime_after, initial_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, runtime_after, initial_edge_weights_tf]))

        # Ensure non-negative runtimes
        runtime_before.assign(tf.maximum(runtime_before, 0.0))
        runtime_after.assign(tf.maximum(runtime_after, 0.0))

        # Record loss and fidelity
        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        # Print every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss_value.numpy():.6f}, Fidelity = {1.0 - loss_value.numpy():.6f}, "
                  f"Initial Weights = [{w1.numpy():.4f}, {w2.numpy():.4f}, {w1.numpy():.4f}], "
                  f"Final Weights = [{w1.numpy():.4f}, {-w2.numpy():.4f}, {w1.numpy():.4f}], "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        # Save perfect solution
        if loss_value == 0:
            solution = {
                "Initial_Weights": [w1.numpy(), w2.numpy(), w1.numpy()],
                "Final_Weights": [w1.numpy(), -w2.numpy(), w1.numpy()],
                "Runtime_Before": runtime_before.numpy(),
                "Runtime_After": runtime_after.numpy(),
                "Fidelity": 1.0
            }
            solutions.append(solution)
            print(f"Solution Found at Step {step}: {solution}")

            # Perturb to search new solutions
            runtime_before.assign(
                tf.maximum(runtime_before + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
            runtime_after.assign(
                tf.maximum(runtime_after + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
            initial_edge_weights_tf.assign(
                initial_edge_weights_tf + tf.random.uniform([2], -3.0, 3.0, dtype=tf.float64))

        # Stagnation handling
        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")
                runtime_before.assign(
                    tf.maximum(runtime_before + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))
                runtime_after.assign(
                    tf.maximum(runtime_after + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))

    # Save all found solutions to CSV
    pd.DataFrame(solutions).to_csv("solutions_n1 to n3_w1 -w2 w1.csv", index=False)

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()

# Initial guesses
fixed_initial_edge_weights = [1.0, 2.0, 1.0]  # [w1, w2, w3] with enforced symmetry w1 == w3
initial_runtime_before = 5.0  # Initial runtime before switching
initial_runtime_after = 5.0   # Initial runtime after switching

# Run optimization
fidelity_simulation_P4_symmetric_weights(
    fixed_initial_edge_weights=fixed_initial_edge_weights,
    initial_runtime_before=initial_runtime_before,
    initial_runtime_after=initial_runtime_after
)
