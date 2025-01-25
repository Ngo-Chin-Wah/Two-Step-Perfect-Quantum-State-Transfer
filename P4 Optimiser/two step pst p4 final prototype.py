import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import logging

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

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """Calculate fidelity using TensorFlow."""
        runtime = tf.cast(runtime, tf.complex128)
        time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

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

    for step in range(500000):
        with tf.GradientTape() as tape:
            # Enforce symmetry for the initial and final edge weights
            initial_edge_weights_symmetric = [initial_edge_weights_tf[0], initial_edge_weights_tf[1], initial_edge_weights_tf[0]]
            final_edge_weights_symmetric = [final_edge_weights_tf[0], final_edge_weights_tf[1], final_edge_weights_tf[0]]

            h_initial = build_hamiltonian_tf(initial_edge_weights_symmetric)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex128)), initial_state_tf)

            h_final = build_hamiltonian_tf(final_edge_weights_symmetric)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_after)

            loss_value = 1.0 - fidelity_value

        gradients = tape.gradient(loss_value, [runtime_before, runtime_after, final_edge_weights_tf, initial_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, runtime_after, final_edge_weights_tf, initial_edge_weights_tf]))

        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        if step % 100 == 0:
            w1, w2 = final_edge_weights_tf.numpy()
            i_w1, i_w2 = initial_edge_weights_tf.numpy()
            print(f"Step {step}: Loss = {loss_value.numpy():.6f}, Fidelity = {1.0 - loss_value.numpy():.6f}, "
                  f"Initial Weights = [{i_w1:.4f}, {i_w2:.4f}, {i_w1:.4f}], "
                  f"Final Weights = [{w1:.4f}, {w2:.4f}, {w1:.4f}], "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_after.numpy():.4f}")

        if loss_value == 0:  # Perfect fidelity achieved
            w1, w2 = final_edge_weights_tf.numpy()
            i_w1, i_w2 = initial_edge_weights_tf.numpy()
            solution = {
                "Initial_Weights": [i_w1, i_w2, i_w1],
                "Final_Weights": [w1, w2, w1],
                "Runtime_Before": runtime_before.numpy(),
                "Runtime_After": runtime_after.numpy(),
                "Fidelity": 1.0 - loss_value.numpy()
            }
            solutions.append(solution)

            print(f"Solution Found: Fidelity = 1.0, Initial Weights = [{i_w1:.4f}, {i_w2:.4f}, {i_w1:.4f}], "
                  f"Final Weights = [{w1:.4f}, {w2:.4f}, {w1:.4f}], Runtime Before = {runtime_before.numpy():.4f}, "
                  f"Runtime After = {runtime_after.numpy():.4f}")

            # Random perturbation
            runtime_before.assign(tf.maximum(runtime_before + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
            runtime_after.assign(tf.maximum(runtime_after + tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64), 0.0))
            initial_edge_weights_tf.assign(initial_edge_weights_tf + tf.random.uniform([2], -0.5, 0.5, dtype=tf.float64))
            final_edge_weights_tf.assign(final_edge_weights_tf + tf.random.uniform([2], -0.5, 0.5, dtype=tf.float64))

        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")

                runtime_before.assign(tf.maximum(runtime_before + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))
                runtime_after.assign(tf.maximum(runtime_after + tf.random.normal([], mean=0.0, stddev=3, dtype=tf.float64), 0.0))
                initial_edge_weights_tf.assign(initial_edge_weights_tf + tf.random.uniform([2], -1, 1, dtype=tf.float64))
                final_edge_weights_tf.assign(final_edge_weights_tf + tf.random.uniform([2], -1, 1, dtype=tf.float64))
            else:
                stagnation_counter = 0

    solutions_df = pd.DataFrame(solutions)
    solutions_df.to_csv("solutions.csv", index=False)

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
