import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings


def fidelity_simulation_P4_fixed_initial_weights(fixed_initial_edge_weights, initial_runtime_before,
                                                 initial_runtime_after, initial_final_edge_weights,
                                                 target_fidelity=0.999, stagnation_threshold=0.0001,
                                                 stagnation_window=200):
    """
    Optimizes the final edge weights and runtimes of a P4 graph Hamiltonian to achieve a target fidelity
    for transferring the quantum state from node 1 to node 4, keeping initial edge weights fixed.

    Parameters:
    fixed_initial_edge_weights (list of float): Fixed edge weights for the Hamiltonian during the first phase.
    initial_runtime_before (float): Initial guess for evolution time before adjustment.
    initial_runtime_after (float): Initial guess for evolution time after adjustment.
    initial_final_edge_weights (list of float): Initial guess for edge weights during the second phase.
    target_fidelity (float): Target fidelity for the quantum state transfer.
    stagnation_threshold (float): Minimum fidelity improvement to avoid stagnation.
    stagnation_window (int): Number of iterations to monitor for stagnation.

    Returns:
    None. Prints details and plots optimization results.
    """

    def build_hamiltonian_tf(edge_weights):
        """Construct the Hamiltonian for a P4 graph using TensorFlow."""
        h = tf.zeros((4, 4), dtype=tf.float32)
        h = tf.tensor_scatter_nd_update(h, [[0, 1], [1, 0]], [edge_weights[0], edge_weights[0]])
        h = tf.tensor_scatter_nd_update(h, [[1, 2], [2, 1]], [edge_weights[1], edge_weights[1]])
        h = tf.tensor_scatter_nd_update(h, [[2, 3], [3, 2]], [edge_weights[2], edge_weights[2]])
        return tf.cast(h, tf.complex64)

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """Calculate fidelity using TensorFlow."""
        runtime = tf.cast(runtime, tf.complex64)  # Ensure runtime is complex64
        time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

    # Define initial and target states for P4
    initial_state = np.zeros((4,), dtype=np.complex64)
    initial_state[0] = 1  # Start at node 1 (index 0)

    target_state = np.zeros((4,), dtype=np.complex64)
    target_state[3] = 1  # Target is node 4 (index 3)

    # Normalize states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Convert to TensorFlow constants
    initial_state_tf = tf.constant(initial_state, dtype=tf.complex64)
    target_state_tf = tf.constant(target_state, dtype=tf.complex64)

    # Convert final edge weights and runtimes to TensorFlow variables for optimization
    final_edge_weights_tf = tf.Variable(initial_final_edge_weights, dtype=tf.float32)
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float32)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float32)

    # Convert fixed initial edge weights to a TensorFlow constant
    initial_edge_weights_tf = tf.constant(fixed_initial_edge_weights, dtype=tf.float32)

    # Set up the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    # Perform optimization
    loss_history = []
    fidelity_history = []
    stagnation_counter = 0

    for step in range(100000):  # Maximum iterations
        with tf.GradientTape() as tape:
            # Calculate the intermediate state using the fixed initial edge weights
            h_initial = build_hamiltonian_tf(initial_edge_weights_tf)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex64)), initial_state_tf)

            # Calculate the final fidelity using the final edge weights
            h_final = build_hamiltonian_tf(final_edge_weights_tf)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_after)

            # Loss function: 1 - fidelity
            loss_value = 1.0 - fidelity_value

        # Compute gradients and apply them
        gradients = tape.gradient(loss_value, [final_edge_weights_tf, runtime_before, runtime_after])
        optimizer.apply_gradients(zip(gradients, [final_edge_weights_tf, runtime_before, runtime_after]))

        # Track loss history and fidelity
        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        # Check for stagnation
        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")
                # Perturb the runtimes and edge weights to escape local minima
                runtime_before.assign(runtime_before + tf.random.normal([], mean=0.1, stddev=0.05))
                runtime_after.assign(runtime_after + tf.random.normal([], mean=0.1, stddev=0.05))
                final_edge_weights_tf.assign(final_edge_weights_tf + tf.random.uniform(final_edge_weights_tf.shape, -1, 1))
            else:
                stagnation_counter = 0  # Reset stagnation counter

        # Print progress
        if step % 100 == 0 or loss_value < 1e-9:
            print(f"Step {step}: Fidelity = {1.0 - loss_value.numpy():.10f}, "
                  f"Final Edge Weights = {final_edge_weights_tf.numpy()}, "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        # Stop if target fidelity is reached
        if (1.0 - loss_value.numpy()) >= target_fidelity:
            break

    # Final results
    final_final_edge_weights = final_edge_weights_tf.numpy()
    final_runtime_before = runtime_before.numpy()
    final_runtime_after = runtime_after.numpy()
    final_fidelity = 1.0 - loss_value.numpy()

    print("\nOptimization Results:")
    print("Fixed Initial Edge Weights:", fixed_initial_edge_weights)
    print("Final Edge Weights (Final):", final_final_edge_weights)
    print("Final Runtime Before Adjustment:", final_runtime_before)
    print("Final Runtime After Adjustment:", final_runtime_after)
    print("Final Fidelity:", final_fidelity)

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()


# Example usage
fixed_initial_edge_weights = [1.0, 1.0, 1.0]  # Fixed initial edge weights
initial_runtime_before = np.pi  # Initial guess for runtime before adjustment
initial_runtime_after = 3.0  # Initial guess for runtime after adjustment
initial_final_edge_weights = [1.0, -1.0, 1.0]  # Initial guess for final edge weights
fidelity_simulation_P4_fixed_initial_weights(fixed_initial_edge_weights, initial_runtime_before, initial_runtime_after,
                                             initial_final_edge_weights)