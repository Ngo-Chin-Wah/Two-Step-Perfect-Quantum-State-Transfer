import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings


def fidelity_simulation_P4_optimized(initial_edge_weights, initial_runtime_before, initial_runtime_after,
                                     target_fidelity=0.9999999):
    """
    Optimizes the edge weights and runtimes of a P4 graph Hamiltonian to achieve a target fidelity
    for transferring the quantum state from node 1 to node 4.

    Parameters:
    initial_edge_weights (list of float): Initial edge weights for the Hamiltonian.
    initial_runtime_before (float): Initial guess for evolution time before adjustment.
    initial_runtime_after (float): Initial guess for evolution time after adjustment.
    target_fidelity (float): Target fidelity for the quantum state transfer.

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

    # Convert edge weights and runtimes to TensorFlow variables for optimization
    edge_weights = tf.Variable(initial_edge_weights, dtype=tf.float32)
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float32)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float32)

    # Set up the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Perform optimization
    loss_history = []
    for step in range(1000):  # Maximum iterations
        with tf.GradientTape() as tape:
            # Calculate the loss (1 - fidelity)
            hamiltonian = build_hamiltonian_tf(edge_weights)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * hamiltonian * tf.cast(runtime_before, tf.complex64)), initial_state_tf)
            fidelity_value = fidelity(hamiltonian, intermediate_state, target_state_tf, runtime_after)
            loss_value = 1.0 - fidelity_value

        # Compute gradients and apply them
        gradients = tape.gradient(loss_value, [edge_weights, runtime_before, runtime_after])
        optimizer.apply_gradients(zip(gradients, [edge_weights, runtime_before, runtime_after]))

        # Track loss history
        loss_history.append(loss_value.numpy())

        # Print progress
        if step % 100 == 0 or loss_value < 1e-9:
            print(
                f"Step {step}: Fidelity = {1.0 - loss_value.numpy():.10f}, Edge Weights = {edge_weights.numpy()}, Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        # Stop if target fidelity is reached
        if (1.0 - loss_value.numpy()) >= target_fidelity:
            break

    # Final results
    final_edge_weights = edge_weights.numpy()
    final_runtime_before = runtime_before.numpy()
    final_runtime_after = runtime_after.numpy()
    final_fidelity = 1.0 - loss_value.numpy()
    h_final = build_hamiltonian_tf(edge_weights).numpy()

    print("\nOptimization Results:")
    print("Initial Edge Weights:", initial_edge_weights)
    print("Initial Runtime Before Adjustment:", initial_runtime_before)
    print("Initial Runtime After Adjustment:", initial_runtime_after)
    print("Final Edge Weights:", final_edge_weights)
    print("Final Runtime Before Adjustment:", final_runtime_before)
    print("Final Runtime After Adjustment:", final_runtime_after)
    print("Final Fidelity:", final_fidelity)
    print("Final Hamiltonian:\n", h_final)

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()


# Example Usage
initial_edge_weights = [3.0, 2.0, 3.0]  # Initial guess for edge weights
initial_runtime_before = 4.0  # Initial guess for runtime before adjustment
initial_runtime_after = 3.0  # Initial guess for runtime after adjustment
fidelity_simulation_P4_optimized(initial_edge_weights, initial_runtime_before, initial_runtime_after)
