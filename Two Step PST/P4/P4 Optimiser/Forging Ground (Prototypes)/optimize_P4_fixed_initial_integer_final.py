import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

# Function to optimize final edge weights and runtimes, keeping initial edge weights fixed
def fidelity_simulation_P4_fixed_initial_weights(fixed_initial_edge_weights, initial_runtime_before,
                                                 initial_runtime_after, initial_final_edge_weights,
                                                 target_fidelity=0.999, stagnation_threshold=0.0001,
                                                 stagnation_window=200):
    """
    Optimize final edge weights and runtimes of a P4 graph Hamiltonian,
    keeping the initial Hamiltonian edge weights fixed, to achieve a target fidelity
    for quantum state transfer from node 1 to node 4.

    Parameters:
    fixed_initial_edge_weights (list of float): Fixed edge weights for the first Hamiltonian phase.
    initial_runtime_before (float): Initial guess for runtime before switching.
    initial_runtime_after (float): Initial guess for runtime after switching.
    initial_final_edge_weights (list of float): Initial guess for final Hamiltonian edge weights.
    target_fidelity (float, optional): Target fidelity to achieve (default 0.999).
    stagnation_threshold (float, optional): Minimum improvement to detect stagnation (default 0.0001).
    stagnation_window (int, optional): Number of steps to monitor stagnation (default 200).

    Returns:
    None: Prints optimization results and plots loss history.
    """
    def build_hamiltonian_tf(edge_weights):
        """
        Build the Hamiltonian matrix for a P4 graph using TensorFlow.

        Parameters:
        edge_weights (list or Tensor): List of edge weights [w1, w2, w3].

        Returns:
        tf.Tensor: 4x4 complex64 Hamiltonian matrix.
        """
        h = tf.zeros((4, 4), dtype=tf.float32)
        h = tf.tensor_scatter_nd_update(h, [[0, 1], [1, 0]], [edge_weights[0], edge_weights[0]])
        h = tf.tensor_scatter_nd_update(h, [[1, 2], [2, 1]], [edge_weights[1], edge_weights[1]])
        h = tf.tensor_scatter_nd_update(h, [[2, 3], [3, 2]], [edge_weights[2], edge_weights[2]])
        return tf.cast(h, tf.complex64)

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """
        Calculate the fidelity between an evolved quantum state and the target state.

        Parameters:
        hamiltonian (tf.Tensor): Hamiltonian matrix.
        initial_state (tf.Tensor): Initial quantum state vector.
        target_state (tf.Tensor): Target quantum state vector.
        runtime (tf.Tensor): Evolution time.

        Returns:
        tf.Tensor: Fidelity value as a scalar Tensor.
        """
        runtime = tf.cast(runtime, tf.complex64)
        time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

    # Define initial and target quantum states
    initial_state = np.zeros((4,), dtype=np.complex64)
    initial_state[0] = 1.0  # Start excitation at node 1

    target_state = np.zeros((4,), dtype=np.complex64)
    target_state[3] = 1.0  # Target excitation at node 4

    # Normalize initial and target states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Convert states to TensorFlow constants
    initial_state_tf = tf.constant(initial_state, dtype=tf.complex64)
    target_state_tf = tf.constant(target_state, dtype=tf.complex64)

    # Initialize trainable variables
    final_edge_weights_tf = tf.Variable(initial_final_edge_weights, dtype=tf.float32)
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float32)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float32)

    # Fixed initial Hamiltonian
    initial_edge_weights_tf = tf.constant(fixed_initial_edge_weights, dtype=tf.float32)

    # Set optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    # Track loss and fidelity history
    loss_history = []
    fidelity_history = []
    stagnation_counter = 0

    # Main optimization loop
    for step in range(100000):
        with tf.GradientTape() as tape:
            # Phase 1 evolution (fixed initial weights)
            h_initial = build_hamiltonian_tf(initial_edge_weights_tf)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex64)),
                initial_state_tf)

            # Phase 2 evolution (trainable final weights)
            h_final = build_hamiltonian_tf(final_edge_weights_tf)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_after)

            loss_value = 1.0 - fidelity_value

        # Apply gradient updates
        gradients = tape.gradient(loss_value, [final_edge_weights_tf, runtime_before, runtime_after])
        optimizer.apply_gradients(zip(gradients, [final_edge_weights_tf, runtime_before, runtime_after]))

        # Record loss and fidelity
        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        # Stagnation detection and perturbation
        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")
                runtime_before.assign(runtime_before + tf.random.normal([], mean=0.1, stddev=0.05))
                runtime_after.assign(runtime_after + tf.random.normal([], mean=0.1, stddev=0.05))
                final_edge_weights_tf.assign(final_edge_weights_tf + tf.random.uniform(final_edge_weights_tf.shape, -1, 1))
            else:
                stagnation_counter = 0

        # Print progress every 100 steps or near-perfect fidelity
        if step % 100 == 0 or loss_value < 1e-9:
            print(f"Step {step}: Fidelity = {1.0 - loss_value.numpy():.10f}, "
                  f"Final Edge Weights = {final_edge_weights_tf.numpy()}, "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        # Stop early if target fidelity achieved
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

    # Plot loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()

# Example usage
fixed_initial_edge_weights = [1.0, 1.0, 1.0]  # Fixed initial edge weights
initial_runtime_before = 5.0  # Initial guess for phase 1 runtime
initial_runtime_after = 5.0  # Initial guess for phase 2 runtime
initial_final_edge_weights = [1.0, -1.0, 1.0]  # Initial guess for final edge weights

# Run optimization
fidelity_simulation_P4_fixed_initial_weights(
    fixed_initial_edge_weights,
    initial_runtime_before,
    initial_runtime_after,
    initial_final_edge_weights
)
