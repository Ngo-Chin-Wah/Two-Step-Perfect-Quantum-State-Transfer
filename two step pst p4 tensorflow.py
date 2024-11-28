import tensorflow as tf
import numpy as np


def fidelity_tf(hamiltonian, initial_state, target_state, time):
    """
    Fidelity calculation using TensorFlow.
    """
    initial_state = initial_state / tf.norm(initial_state)
    target_state = target_state / tf.norm(target_state)
    time_evolution_operator = tf.linalg.expm(-1j * hamiltonian * time)
    evolved_state = tf.matmul(time_evolution_operator, initial_state)
    fidelity_value = tf.abs(tf.linalg.adjoint(target_state) @ evolved_state) ** 2
    return tf.squeeze(fidelity_value)


def construct_hamiltonian(base_hamiltonian, edge_weights, edges):
    """
    Constructs a Hamiltonian matrix with given edge weights.

    Parameters:
    - base_hamiltonian: The base Hamiltonian structure (as a tensor).
    - edge_weights: Tensor of edge weights to apply.
    - edges: List of edge tuples specifying connections in the graph.

    Returns:
    - Hamiltonian matrix with updated edge weights.
    """
    hamiltonian = tf.identity(base_hamiltonian)
    for i, (u, v) in enumerate(edges):
        weight = edge_weights[i]
        hamiltonian = tf.tensor_scatter_nd_update(
            hamiltonian,
            [[u, v], [v, u]],
            [weight, weight]
        )
    return hamiltonian


def optimize_with_tensorflow(
        hamiltonian, initial_state, target_state, learning_rate=0.01, epochs=1000, fidelity_tolerance=0.999999
):
    """
    Use TensorFlow to optimize t1, t2, and edge weights.
    """
    # Convert tensors to complex128 for compatibility
    hamiltonian = tf.cast(hamiltonian, dtype=tf.complex128)
    initial_state = tf.cast(initial_state, dtype=tf.complex128)
    target_state = tf.cast(target_state, dtype=tf.complex128)

    # Define variables
    t1 = tf.Variable(1.0, dtype=tf.float64, name="t1")
    t2 = tf.Variable(1.0, dtype=tf.float64, name="t2")
    edge_weights_init = tf.Variable([1.0, 1.0, 1.0], dtype=tf.float64, name="edge_weights_init")
    edge_weights_adj = tf.Variable([1.0, 1.0, 1.0], dtype=tf.float64, name="edge_weights_adj")

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Define optimization step
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            # Define the edges of the graph
            edges = [(0, 1), (1, 2), (2, 3)]

            # Construct Hamiltonians
            h_init = construct_hamiltonian(hamiltonian, edge_weights_init, edges)
            h_adj = construct_hamiltonian(hamiltonian, edge_weights_adj, edges)

            # Fidelity before adjustment
            intermediate_state = tf.linalg.expm(-1j * h_init * tf.cast(t1, tf.complex128)) @ initial_state
            # Fidelity after adjustment
            final_state = tf.linalg.expm(-1j * h_adj * tf.cast(t2, tf.complex128)) @ intermediate_state
            fidelity_value = tf.abs(tf.linalg.adjoint(target_state) @ final_state) ** 2

            # Loss to minimize
            loss = 1 - fidelity_value

        gradients = tape.gradient(loss, [t1, t2, edge_weights_init, edge_weights_adj])
        optimizer.apply_gradients(zip(gradients, [t1, t2, edge_weights_init, edge_weights_adj]))
        return fidelity_value

    # Training loop
    for epoch in range(epochs):
        fidelity_value = step()
        if fidelity_value >= fidelity_tolerance:
            break

    # Results
    return {
        "t1": t1.numpy(),
        "t2": t2.numpy(),
        "initial_edge_weights": edge_weights_init.numpy(),
        "adjusted_edge_weights": edge_weights_adj.numpy(),
        "fidelity": fidelity_value.numpy(),
    }


# Example usage
if __name__ == "__main__":
    # P4 adjacency matrix (Hamiltonian)
    hamiltonian = tf.constant([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=tf.float64)

    # Initial and target states
    initial_state = tf.constant([[1.0], [0.0], [0.0], [0.0]], dtype=tf.float64)
    target_state = tf.constant([[0.0], [0.0], [0.0], [1.0]], dtype=tf.float64)

    # Optimize with TensorFlow
    result = optimize_with_tensorflow(
        hamiltonian, initial_state, target_state, learning_rate=0.01, epochs=1000
    )
    print(result)
