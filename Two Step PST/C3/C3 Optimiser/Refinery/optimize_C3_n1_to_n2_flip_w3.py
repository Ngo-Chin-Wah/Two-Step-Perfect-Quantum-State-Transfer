import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

# Function to optimize runtimes and edge weights of a C3 graph to achieve target fidelity
def fidelity_simulation_C3_weights(fixed_initial_edge_weights, initial_runtime_before,
                                   initial_runtime_after, target_fidelity=0.999999999999,
                                   stagnation_threshold=0.1, stagnation_window=12000):
    """
    Optimize the runtimes and edge weights of a C3 graph Hamiltonian to achieve a target fidelity
    for quantum state transfer from node 1 to node 2.

    Parameters:
    fixed_initial_edge_weights (list): Initial guess for edge weights [fixed_w1, guess_w2, guess_w3].
    initial_runtime_before (float): Initial guess for runtime before state adjustment.
    initial_runtime_after (float): Initial guess for runtime after state adjustment (not actively used).
    target_fidelity (float, optional): Desired final fidelity (default 0.999999999999).
    stagnation_threshold (float, optional): Threshold for stagnation detection (default 0.1).
    stagnation_window (int, optional): Number of steps to consider for stagnation (default 12000).

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

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """
        Compute the fidelity between an evolved state and a target state using TensorFlow.

        Parameters:
        hamiltonian (tf.Tensor): Hamiltonian matrix for evolution.
        initial_state (tf.Tensor): Initial quantum state vector.
        target_state (tf.Tensor): Target quantum state vector.
        runtime (tf.Tensor): Evolution time.

        Returns:
        tf.Tensor: Fidelity value as a scalar Tensor.
        """
        runtime = tf.cast(runtime, tf.complex128)
        time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

    # Define initial and target quantum states
    initial_state = np.array([1, 0, 0], dtype=np.complex128)
    target_state = np.array([0, 1, 0], dtype=np.complex128)

    # Convert initial and target states to TensorFlow tensors
    initial_state_tf = tf.constant(initial_state, dtype=tf.complex128)
    target_state_tf = tf.constant(target_state, dtype=tf.complex128)

    # Initialize runtime variable
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float64)

    # Initialize variable edge weights (w2, w3); w1 is fixed at 1.0
    initial_edge_weights_tf = tf.Variable([fixed_initial_edge_weights[1], fixed_initial_edge_weights[2]], dtype=tf.float64)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)  # Adam optimizer

    loss_history, fidelity_history, solutions = [], [], []  # Record histories
    stagnation_counter = 0  # Counter for stagnation events

    # Main optimization loop
    for step in range(500000):
        with tf.GradientTape() as tape:
            w1 = 1.0  # Fixed first edge weight
            w2 = initial_edge_weights_tf[0]  # Variable second edge weight
            w3 = initial_edge_weights_tf[1]  # Variable third edge weight

            initial_weights = [w1, w2, w3]  # Initial Hamiltonian weights
            final_weights = [w1, w2, -w3]  # Final Hamiltonian weights after flipping w3

            h_initial = build_hamiltonian_tf(initial_weights)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex128)), initial_state_tf)

            h_final = build_hamiltonian_tf(final_weights)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_before)

            loss_value = 1.0 - fidelity_value  # Loss is (1 - fidelity)

        # Apply gradients to runtime and edge weights
        gradients = tape.gradient(loss_value, [runtime_before, initial_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, initial_edge_weights_tf]))

        # Ensure runtime stays non-negative
        runtime_before.assign(tf.maximum(runtime_before, 0.0))

        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        # Print intermediate progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss_value.numpy():.6f}, Fidelity = {1.0 - loss_value.numpy():.6f}, "
                  f"Initial Weights = [{w1:.4f}, {w2.numpy():.4f}, {w3.numpy():.4f}], "
                  f"Final Weights = [{w1:.4f}, {w2.numpy():.4f}, {-w3.numpy():.4f}], "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_before.numpy():.4f}")

        # Save solution if target fidelity is achieved
        if loss_value < (1 - target_fidelity):
            solution = {
                "Initial_Weights": [w1, w2.numpy(), w3.numpy()],
                "Final_Weights": [w1, w2.numpy(), -w3.numpy()],
                "Runtime_Before": runtime_before.numpy(),
                "Runtime_After": runtime_before.numpy(),
                "Fidelity": 1.0
            }
            solutions.append(solution)
            print(f"Solution Found at Step {step}: {solution}")

            # Perturb runtime and weights after finding a solution
            while True:
                runtime_before.assign(
                    tf.maximum(0.5 + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
                if runtime_before.numpy() <= 5.0:
                    break

            initial_edge_weights_tf.assign(
                3.0 + tf.random.uniform([2], -3.0, 3.0, dtype=tf.float64))

        # Detect stagnation and perturb if needed
        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")
                while True:
                    initial_edge_weights_tf.assign(
                        3.0 + tf.random.uniform([2], -3.0, 3.0, dtype=tf.float64))
                    runtime_before.assign(
                        tf.maximum(0.5 + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))
                    if runtime_before.numpy() <= 5.0:
                        break

    # Save all found solutions to CSV file
    pd.DataFrame(solutions).to_csv("solutions_n1_to_n2_C3_1_w2_-w3_0.999999999999.csv", index=False)

    # Plot loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()

# Define initial parameters and run optimization
fixed_initial_edge_weights = [1.0, 1.0, 1.0]  # initial edge weights: [fixed (ignored), guess for w2, guess for w3]
initial_runtime_before = 1.0  # Initial runtime before the state transfer
initial_runtime_after = 1.0   # Initial runtime after the state transfer

fidelity_simulation_C3_weights(
    fixed_initial_edge_weights=fixed_initial_edge_weights,
    initial_runtime_before=initial_runtime_before,
    initial_runtime_after=initial_runtime_after
)