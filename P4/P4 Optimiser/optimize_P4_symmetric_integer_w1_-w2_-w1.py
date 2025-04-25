import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

# Function to optimize runtimes and symmetric integer edge weights of a P4 graph
def fidelity_simulation_P4_symmetric_weights(fixed_initial_edge_weights, initial_runtime_before,
                                             initial_runtime_after, initial_final_edge_weights,
                                             target_fidelity=1.0, stagnation_threshold=0.0001,
                                             stagnation_window=1000):
    """
    Optimize runtimes and symmetric integer edge weights of a P4 graph Hamiltonian
    to achieve a target fidelity for quantum state transfer from node 1 to node 4.

    In this version, edge weights are enforced to be integers and symmetric (w1 = w3).

    Parameters:
    fixed_initial_edge_weights (list of int): Fixed edge weights for initial Hamiltonian phase (symmetric).
    initial_runtime_before (float): Initial guess for evolution time before adjustment.
    initial_runtime_after (float): Initial guess for evolution time after adjustment.
    initial_final_edge_weights (list of int): Initial guess for edge weights after switching (symmetric).
    target_fidelity (float, optional): Target fidelity to reach (default 1.0).
    stagnation_threshold (float, optional): Minimum fidelity improvement to avoid stagnation (default 0.0001).
    stagnation_window (int, optional): Number of iterations to check for stagnation (default 1000).

    Returns:
    None: Prints optimization results and plots loss history.
    """
    # Validate symmetry and integer constraint
    if not (all(isinstance(w, int) for w in fixed_initial_edge_weights) and
            fixed_initial_edge_weights[0] == fixed_initial_edge_weights[2]):
        raise ValueError("fixed_initial_edge_weights must be integers and symmetric (w1 == w3).")
    if not (all(isinstance(w, int) for w in initial_final_edge_weights) and
            initial_final_edge_weights[0] == initial_final_edge_weights[2]):
        raise ValueError("initial_final_edge_weights must be integers and symmetric (w1 == w3).")

    def build_hamiltonian_tf(edge_weights):
        """
        Build the Hamiltonian matrix for a symmetric P4 graph using TensorFlow.

        Parameters:
        edge_weights (list or Tensor): Edge weights [w1, w2] for edges (0-1), (1-2), with w3 = w1 enforced.

        Returns:
        tf.Tensor: 4x4 complex64 Hamiltonian matrix.
        """
        h = tf.zeros((4, 4), dtype=tf.float32)
        w1 = edge_weights[0]
        w2 = edge_weights[1]
        w3 = w1  # Enforce symmetry

        h = tf.tensor_scatter_nd_update(h, [[0, 1], [1, 0]], [w1, w1])
        h = tf.tensor_scatter_nd_update(h, [[1, 2], [2, 1]], [w2, w2])
        h = tf.tensor_scatter_nd_update(h, [[2, 3], [3, 2]], [w3, w3])
        return tf.cast(h, tf.complex64)

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """
        Calculate the fidelity between an evolved state and a target state using TensorFlow.

        Parameters:
        hamiltonian (tf.Tensor): Hamiltonian matrix for evolution.
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

    # Define initial and target states
    initial_state = np.zeros((4,), dtype=np.complex64)
    initial_state[0] = 1.0  # Start at node 1 (index 0)
    target_state = np.zeros((4,), dtype=np.complex64)
    target_state[2] = 1.0  # Target node 3 (index 2)

    # Normalize states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Convert states to TensorFlow constants
    initial_state_tf = tf.constant(initial_state, dtype=tf.complex64)
    target_state_tf = tf.constant(target_state, dtype=tf.complex64)

    # Initialize trainable variables
    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float32)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float32)
    final_edge_weights_tf = tf.Variable(initial_final_edge_weights[:2], dtype=tf.float32)  # w1 and w2

    # Fixed initial edge weights (TensorFlow constant)
    initial_edge_weights_tf = tf.constant(fixed_initial_edge_weights[:2], dtype=tf.float32)

    # Optimizer setup
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    # History trackers
    loss_history = []
    fidelity_history = []
    stagnation_counter = 0

    # Main optimization loop
    for step in range(200000):
        with tf.GradientTape() as tape:
            final_edge_weights_tf.assign(tf.round(final_edge_weights_tf))  # Enforce integer constraint

            # Phase 1: evolve using fixed initial edge weights
            h_initial = build_hamiltonian_tf(initial_edge_weights_tf)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex64)), initial_state_tf)

            # Phase 2: evolve using optimized final edge weights
            h_final = build_hamiltonian_tf(final_edge_weights_tf)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_after)

            # Loss function
            loss_value = 1.0 - fidelity_value

        # Apply gradients
        gradients = tape.gradient(loss_value, [runtime_before, runtime_after, final_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, runtime_after, final_edge_weights_tf]))

        # Round final weights after update
        final_edge_weights_tf.assign(tf.round(final_edge_weights_tf))

        # Record loss and fidelity
        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        # Check for stagnation
        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")

                # Perturb runtimes
                runtime_before.assign(np.abs(runtime_before + tf.random.normal([], mean=0.0, stddev=3)))
                runtime_after.assign(np.abs(runtime_after + tf.random.normal([], mean=0.0, stddev=3)))

                # Perturb symmetric edge weights
                perturbation = tf.random.uniform([], minval=-1, maxval=1)
                final_edge_weights_tf.assign([
                    final_edge_weights_tf[0] + perturbation,
                    final_edge_weights_tf[1] + tf.random.uniform([], -1, 1)
                ])
            else:
                stagnation_counter = 0  # Reset if improved

        # Print progress
        if step % 100 == 0 or loss_value < 1e-9:
            w1, w2 = final_edge_weights_tf.numpy()
            print(f"Step {step}: Fidelity = {1.0 - loss_value.numpy():.10f}, "
                  f"Final Edge Weights = [{w1:.0f}, {w2:.0f}, {w1:.0f}], "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        # Early stopping if target fidelity is reached
        if (1.0 - loss_value.numpy()) == target_fidelity:
            break

    # Final results
    final_final_edge_weights = tf.round(final_edge_weights_tf).numpy()
    final_runtime_before = runtime_before.numpy()
    final_runtime_after = runtime_after.numpy()
    final_fidelity = 1.0 - loss_value.numpy()

    print("\nOptimization Results:")
    print("Fixed Initial Edge Weights:", fixed_initial_edge_weights)
    print("Final Edge Weights (Final):", [final_final_edge_weights[0], final_final_edge_weights[1], final_final_edge_weights[0]])
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

# Define initial settings and run the optimization
fixed_initial_edge_weights = [1, 2, 1]  # Fixed symmetric edge weights
initial_runtime_before = 5.0  # Initial runtime guess before adjustment
initial_runtime_after = 5.0   # Initial runtime guess after adjustment
initial_final_edge_weights = [-1, 2, -1]  # Initial symmetric guess for final edge weights

# Run simulation
fidelity_simulation_P4_symmetric_weights(
    fixed_initial_edge_weights=fixed_initial_edge_weights,
    initial_runtime_before=initial_runtime_before,
    initial_runtime_after=initial_runtime_after,
    initial_final_edge_weights=initial_final_edge_weights
)
)