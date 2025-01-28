import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Mute TensorFlow warnings

def fidelity_simulation_P4_symmetric_weights(fixed_initial_edge_weights, initial_runtime_before,
                                             initial_runtime_after, target_fidelity=0.999999999999,
                                             stagnation_threshold=0.1, stagnation_window=7000):
    """
    Optimizes runtimes of a P4 graph Hamiltonian to achieve a target fidelity
    for transferring the quantum state from node 1 to node 4, prioritizing symmetric graphs.
    Initial edge weights are optimized, and final edge weights are set to [-w1, w2, -w1].
    """

    def build_hamiltonian_tf(edge_weights):
        """Construct the Hamiltonian for a symmetric P4 graph using TensorFlow."""
        edge_weights = tf.convert_to_tensor(edge_weights, dtype=tf.float64)
        h = tf.stack([[0, edge_weights[0], 0, 0],
                      [edge_weights[0], 0, edge_weights[1], 0],
                      [0, edge_weights[1], 0, edge_weights[2]],
                      [0, 0, edge_weights[2], 0]])
        return tf.cast(h, tf.complex128)

    def fidelity(hamiltonian, initial_state, target_state, runtime):
        """Calculate fidelity using TensorFlow."""
        runtime = tf.cast(runtime, tf.complex128)
        time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * runtime)
        evolved_state = tf.linalg.matvec(time_evolution_operator, initial_state)
        fidelity_value = tf.abs(tf.reduce_sum(tf.math.conj(target_state) * evolved_state)) ** 2
        return fidelity_value

    initial_state = np.array([1, 0, 0, 0], dtype=np.complex128)
    target_state = np.array([0, 0, 0, 1], dtype=np.complex128)

    initial_state_tf = tf.constant(initial_state, dtype=tf.complex128)
    target_state_tf = tf.constant(target_state, dtype=tf.complex128)

    runtime_before = tf.Variable(initial_runtime_before, dtype=tf.float64)

    # Fix w1 = 1 and optimize only w2
    initial_edge_weights_tf = tf.Variable([fixed_initial_edge_weights[1]], dtype=tf.float64)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    loss_history, fidelity_history, solutions = [], [], []
    stagnation_counter = 0

    for step in range(500000):
        with tf.GradientTape() as tape:
            w1 = 1.0  # Fixed value
            w2 = initial_edge_weights_tf[0]

            # Enforce symmetry: w3 = w1
            initial_weights = [w1, w2, w1]

            # Final weights set to [-w1, w2, -w1]
            final_weights = [-w1, w2, -w1]

            h_initial = build_hamiltonian_tf(initial_weights)
            intermediate_state = tf.linalg.matvec(
                tf.linalg.expm(-1j * h_initial * tf.cast(runtime_before, tf.complex128)), initial_state_tf)

            h_final = build_hamiltonian_tf(final_weights)
            fidelity_value = fidelity(h_final, intermediate_state, target_state_tf, runtime_before)

            loss_value = 1.0 - fidelity_value

        gradients = tape.gradient(loss_value, [runtime_before, initial_edge_weights_tf])
        optimizer.apply_gradients(zip(gradients, [runtime_before, initial_edge_weights_tf]))

        # Ensure runtimes remain non-negative
        runtime_before.assign(tf.maximum(runtime_before, 0.0))

        loss_history.append(loss_value.numpy())
        fidelity_history.append(1.0 - loss_value.numpy())

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss_value.numpy():.6f}, Fidelity = {1.0 - loss_value.numpy():.6f}, "
                  f"Initial Weights = [{w1:.4f}, {w2.numpy():.4f}, {w1:.4f}], "
                  f"Final Weights = [{-w1:.4f}, {w2.numpy():.4f}, {-w1:.4f}], "
                  f"Runtime Before = {runtime_before.numpy():.4f}, Runtime After = {runtime_before.numpy():.4f}")

        if loss_value < (1 - target_fidelity):
            solution = {
                "Initial_Weights": [w1, w2.numpy(), w1],
                "Final_Weights": [-w1, w2.numpy(), -w1],
                "Runtime_Before": runtime_before.numpy(),
                "Runtime_After": runtime_before.numpy(),
                "Fidelity": 1.0
            }
            solutions.append(solution)
            print(f"Solution Found at Step {step}: {solution}")

            # Perturb runtime_before with a valid value
            while True:
                runtime_before.assign(
                    tf.maximum(1 + tf.random.normal([], mean=3.0, stddev=2.0, dtype=tf.float64), 0.0))
                if 0.1 <= runtime_before.numpy() <= 10.0:
                    break  # Exit the loop if runtime is valid

            # Perturb edge weights as before
            initial_edge_weights_tf.assign(initial_edge_weights_tf + tf.random.uniform([1], -5.0, 5.0, dtype=tf.float64))

        if step >= stagnation_window:
            recent_fidelities = fidelity_history[-stagnation_window:]
            improvement = max(recent_fidelities) - min(recent_fidelities)

            if improvement < stagnation_threshold:
                stagnation_counter += 1
                print(f"Stagnation detected at step {step}. Perturbing variables.")
                # Perturb runtime_before with a valid value
                while True:
                    runtime_before.assign(
                        tf.maximum(1 + tf.random.normal([], mean=3.0, stddev=2.0, dtype=tf.float64), 0.0))
                    if 0.1 <= runtime_before.numpy() <= 10.0:
                        break  # Exit the loop if runtime is valid
                    initial_edge_weights_tf.assign(initial_edge_weights_tf + tf.random.uniform([1], -5.0, 5.0, dtype=tf.float64))

    pd.DataFrame(solutions).to_csv("solutions_n1 to n4_-1 w2 -1_ 0.999999999999(2).csv", index=False)

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (1 - Fidelity)")
    plt.title("Loss Over Optimization Steps")
    plt.grid()
    plt.show()


fixed_initial_edge_weights = [1.0, -2.5, 1.0]  # Initial guess for edge weights (w1, w2, w3 where w3 = w1)
initial_runtime_before = 2.0  # Initial runtime before the state transfer
initial_runtime_after = 2.0   # Initial runtime after the state transfer

fidelity_simulation_P4_symmetric_weights(
    fixed_initial_edge_weights=fixed_initial_edge_weights,
    initial_runtime_before=initial_runtime_before,
    initial_runtime_after=initial_runtime_after
)
