import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

# Function to optimize independent initial and final edge weights and runtimes of a C3 graph
def fidelity_simulation_C3_individual_weights(initial_edge_weights, final_edge_weights,
                                              initial_runtime_before, initial_runtime_after,
                                              target_fidelity=0.999, stagnation_threshold=0.1,
                                              stagnation_window=1500):
    """
    Optimize the Hamiltonians and runtimes of a C3 graph to achieve a target fidelity
    for quantum state transfer from node 1 to node 3.

    In this version, all three initial and final edge weights are independent,
    and the two runtimes are also independently optimized.

    Parameters:
    initial_edge_weights (list): Initial guess for the three initial Hamiltonian edge weights.
    final_edge_weights (list): Initial guess for the three final Hamiltonian edge weights.
    initial_runtime_before (float): Initial guess for runtime before state adjustment.
    initial_runtime_after (float): Initial guess for runtime after state adjustment.
    target_fidelity (float, optional): Desired final fidelity (default 0.999).
    stagnation_threshold (float, optional): Threshold for stagnation detection (default 0.1).
    stagnation_window (int, optional): Number of steps to consider for stagnation (default 1500).

    Returns:
    None: Saves optimized solutions to a CSV file and plots loss history.
    """

    def build_hamiltonian_tf(edge_weights):
        """
        Build the Hamiltonian matrix for a C3 graph using TensorFlow.

        Parameters:
        edge_weights (list or Tensor): List or Tensor of edge weights [w1, w2, w3].

        Returns:
        tf.Tensor: 3x3 complex128 Hamiltonian matrix.
        """
        edge_weights = tf.convert_to_tensor(edge_weights, dtype=tf.float64)
        h = tf.stack([[0, edge_weights[0], edge_weights[2]],
                      [edge_weights[0], 0, edge_weights[1]],
                      [edge_weights[2], edge_weights[1], 0]])
        return tf.cast(h, tf.complex128)

    # Define initial and target quantum states
    initial_state = np.array([1, 0, 0], dtype=np.complex128)
    target_state = np.array([0, 0, 1], dtype=np.complex128)

    # Convert initial and target states to TensorFlow tensors
    initial_state_tf = tf.constant(initial_state, dtype=tf.complex128)
    target_state_tf = tf.constant(target_state, dtype=tf.complex128)

    # Initialize runtime variables
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float64)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float64)

    # Initialize independent initial and final edge weight variables
    initial_edge_weights_tf = tf.Variable(initial_edge_weights, dtype=tf.float64)
    final_edge_weights_tf = tf.Variable(final_edge_weights, dtype=tf.float64)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)  # Adam optimizer

    loss_history, fidelity_history, solutions = [], [], []  # Record histories
    stagnation_counter = 0  # Counter for stagnation events

    # Main optimization loop
    for step in range(500000):
        with tf.GradientTape() as tape:
            h_initial = build_hamiltonian_tf(initial_edge_weights_tf)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex128)),
                initial_state_tf)

            h_final = build_hamiltonian_tf(final_edge_weights_tf)
            final_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_final * tf.cast(runtime_after, tf.complex128)),
                intermediate_state)

            fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state_tf) * final_state)) ** 2
            loss_value = 1.0 - fidelity_value  # Loss is (1 - fidelity)

        # Apply gradients to all variables
        gradients = tape.gradient(loss_value, [runtime_before, runtime_after,
                                               initial_edge_weights_tf, final_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, runtime_after,
                                                  initial_edge_weights_tf, final_edge_weights_tf]))

        # Ensure runtimes stay non-negative
        runtime_before.assign(tf.maximum(runtime_before, 0.0))
        runtime_after.assign(tf.maximum(runtime_after, 0.0))

        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        # Print intermediate progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss_value.numpy():.6f}, Fidelity = {1.0 - loss_value.numpy():.6f}, "
                  f"Initial Weights = {initial_edge_weights_tf.numpy()}, "
                  f"Final Weights = {final_edge_weights_tf.numpy()}, "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        # Save solution if target fidelity is achieved
        if loss_value < (1 - target_fidelity):
            solution = {
                "Initial_Weights": initial_edge_weights_tf.numpy().tolist(),
                "Final_Weights": final_edge_weights_tf.numpy().tolist(),
                "Runtime_Before": runtime_before.numpy(),
                "Runtime_After": runtime_after.numpy(),
                "Fidelity": 1.0
            }
            solutions.append(solution)
            print(f"Solution Found at Step {step}: {solution}")

            # Perturb runtimes slightly to continue search
            while True:
                runtime_before.assign(
                    tf.maximum(runtime_before + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
                runtime_after.assign(
                    tf.maximum(runtime_after + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
                if runtime_before.numpy() <= 5.0 and runtime_after.numpy() <= 5.0:
                    break

            # Perturb both initial and final edge weights
            initial_edge_weights_tf.assign(
                initial_edge_weights_tf + tf.random.uniform([3], -3.0, 3.0, dtype=tf.float64))
            final_edge_weights_tf.assign(
                final_edge_weights_tf + tf.random.uniform([3], -3.0, 3.0, dtype=tf.float64))

        # Detect stagnation and perturb if needed
        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)
            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")
                while True:
                    runtime_before.assign(
                        tf.maximum(runtime_before + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))
                    runtime_after.assign(
                        tf.maximum(runtime_after + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))
                    if runtime_before.numpy() <= 5.0 and runtime_after.numpy() <= 5.0:
                        break

    # Save all found solutions to CSV file
    pd.DataFrame(solutions).to_csv("solutions_C3_individual_weights.csv", index=False)

    # Plot loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()

# Define initial parameters and run optimization
initial_edge_weights = [1.0, 0.5, 1.0]  # Initial guess for initial edge weights
final_edge_weights = [-1.0, 0.5, -1.0]  # Initial guess for final edge weights
initial_runtime_before = 1.0  # Initial runtime before the state transfer
initial_runtime_after = 1.0  # Initial runtime after the state transfer

fidelity_simulation_C3_individual_weights(
    initial_edge_weights=initial_edge_weights,
    final_edge_weights=final_edge_weights,
    initial_runtime_before=initial_runtime_before,
    initial_runtime_after=initial_runtime_after
)
