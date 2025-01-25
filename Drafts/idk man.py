import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import logging
from scipy.linalg import expm  # High-precision matrix exponentiation

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

def fidelity_simulation_P4_symmetric_weights(fixed_initial_edge_weights, initial_runtime_before,
                                             initial_runtime_after, initial_final_edge_weights,
                                             target_fidelity=1.0, stagnation_threshold=0.0001,
                                             stagnation_window=1000):
    """
    Optimizes runtimes of a P4 graph Hamiltonian to achieve a target fidelity
    for transferring the quantum state from node 1 to node 4, prioritizing symmetric graphs.
    """

    def build_hamiltonian_tf(edge_weights):
        """Construct the Hamiltonian for a symmetric P4 graph using TensorFlow."""
        h = tf.zeros((4, 4), dtype=tf.float64)
        w1, w2 = edge_weights[0], edge_weights[1]
        w3 = w1  # Symmetry enforced

        h = tf.tensor_scatter_nd_update(h, [[0, 1], [1, 0]], [w1, w1])
        h = tf.tensor_scatter_nd_update(h, [[1, 2], [2, 1]], [w2, w2])
        h = tf.tensor_scatter_nd_update(h, [[2, 3], [3, 2]], [w3, w3])
        return tf.cast(h, tf.complex128)

    def high_precision_expm_tf(hamiltonian, runtime):
        """Use TensorFlow's expm to ensure gradients are tracked."""
        return tf.linalg.expm(-1j * hamiltonian * tf.cast(runtime, tf.complex128))

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """Calculate fidelity using high-precision expm."""
        runtime = tf.cast(runtime, tf.complex128)
        time_evolution_operator = high_precision_expm_tf(hamiltonian, runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

    def validate_exact_fidelity(hamiltonian, initial_state, target_state, runtime):
        time_evolution_operator = expm(-1j * hamiltonian * runtime.numpy())
        evolved_state = np.dot(time_evolution_operator, initial_state)
        fidelity_value = np.abs(np.vdot(target_state, evolved_state)) ** 2
        return np.isclose(fidelity_value, 1.0, atol=1e-14)

    initial_state = np.zeros((4,), dtype=np.complex128)
    initial_state[0] = 1  # Start at node 1 (index 0)

    target_state = np.zeros((4,), dtype=np.complex128)
    target_state[2] = 1  # Target is node 3 (index 2)

    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    initial_state_tf = tf.constant(initial_state, dtype=tf.complex128)
    target_state_tf = tf.constant(target_state, dtype=tf.complex128)

    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float64)
    runtime_after = tf.Variable(initial_runtime_after, dtype=tf.float64)
    final_edge_weights_tf = tf.Variable(initial_final_edge_weights[:2], dtype=tf.float64)

    # Make the initial edge weights trainable
    initial_edge_weights_tf = tf.Variable(fixed_initial_edge_weights[:2], dtype=tf.float64)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    loss_history = []
    fidelity_history = []
    solutions = []
    stagnation_counter = 0

    for step in range(100000):
        with tf.GradientTape() as tape:
            # Enforce symmetry for the initial and final edge weights
            initial_edge_weights_symmetric = [initial_edge_weights_tf[0], initial_edge_weights_tf[1], initial_edge_weights_tf[0]]
            final_edge_weights_symmetric = [final_edge_weights_tf[0], final_edge_weights_tf[1], final_edge_weights_tf[0]]

            h_initial = build_hamiltonian_tf(initial_edge_weights_symmetric)
            intermediate_state = tf.linalg.matvec(
                high_precision_expm_tf(h_initial, runtime_before), initial_state_tf)

            h_final = build_hamiltonian_tf(final_edge_weights_symmetric)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_after)

            loss_value = 1.0 - fidelity_value

        gradients = tape.gradient(loss_value, [runtime_before, runtime_after, final_edge_weights_tf, initial_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, runtime_after, final_edge_weights_tf, initial_edge_weights_tf]))

        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        if loss_value == 0:
            h_final_numpy = h_final.numpy()
            numpy_fidelity = validate_exact_fidelity(h_final_numpy, intermediate_state.numpy(), target_state, runtime_after.numpy())

            if numpy_fidelity:
                print(f"Perfect Solution Verified at Step {step}")
                solutions.append({
                    "Initial_Weights": initial_edge_weights_tf.numpy(),
                    "Final_Weights": final_edge_weights_tf.numpy(),
                    "Runtime_Before": runtime_before.numpy(),
                    "Runtime_After": runtime_after.numpy(),
                    "Fidelity": 1.0
                })

    solutions_df = pd.DataFrame(solutions)
    solutions_df.to_csv("solutions2.csv", index=False)

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()

fixed_initial_edge_weights = [1.0, 2.0, 1.0]
initial_runtime_before = 5.0
initial_runtime_after = 5.0
initial_final_edge_weights = [-1.0, 2.0, -1.0]

fidelity_simulation_P4_symmetric_weights(
    fixed_initial_edge_weights=fixed_initial_edge_weights,
    initial_runtime_before=initial_runtime_before,
    initial_runtime_after=initial_runtime_after,
    initial_final_edge_weights=initial_final_edge_weights)
