import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    time_evolution_operator = expm(-1j * hamiltonian * time)
    evolved_state = np.dot(time_evolution_operator, initial_state)
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2
    return fidelity, evolved_state

def generate_p4_hamiltonian(weights):
    size = 4
    h = np.zeros((size, size))
    h[0, 1] = h[1, 0] = weights[0]
    h[1, 2] = h[2, 1] = weights[1]
    h[2, 3] = h[3, 2] = weights[2]
    return h

if __name__ == "__main__":
    # --- Hard-coded runtimes (only ONE set) ---
    run_time_1 = (7**0.5/4)*np.pi
    run_time_2 = (7**0.5/4)*np.pi+3

    # Define Hamiltonians
    H1 = generate_p4_hamiltonian([1.0, 6/(7)**0.5, 1.0])
    H2 = generate_p4_hamiltonian([-1.0, 6/(7)**0.5, -1.0])

    # Initial state (excitation at node 1)
    initial_state = np.array([1, 0, 0, 0])[:, np.newaxis]

    # Target state (excitation at node 4)
    target_state = np.array([0, 0, 0, 1])[:, np.newaxis]

    # Normalize states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Phase 1: Fidelity and intermediate state
    fidelity_phase1, intermediate_state = calculate_fidelity(
        H1, initial_state, target_state, run_time_1
    )

    # Phase 2: Fidelity using the intermediate state
    fidelity_phase2, _ = calculate_fidelity(
        H2, intermediate_state, target_state, run_time_2
    )

    # Total fidelity
    total_fidelity = fidelity_phase2
    print(f"Fidelity after full runtime: {total_fidelity:.30f}")

    # -------------------
    # PLOT PROBABILITY AT NODE 4 VS TIME
    # -------------------
    total_time = run_time_1 + run_time_2
    time_points = np.linspace(0, total_time, 300)
    prob_node_4 = []

    for t in time_points:
        if t <= run_time_1:
            # During first phase
            U_t = expm(-1j * H1 * t)
            state_t = np.dot(U_t, initial_state)
        else:
            # During second phase
            U_t = expm(-1j * H2 * (t - run_time_1))
            state_t = np.dot(U_t, intermediate_state)
        # Probability at node 4
        prob_node_4.append(np.abs(state_t[3])**2)

    # Probability exactly at the switching time
    switch_probability = np.abs(
        (expm(-1j * H1 * run_time_1).dot(initial_state))[3]
    )**2

    # Plot
    plt.plot(time_points, prob_node_4)
    # Red dot at switching point
    # plt.scatter(run_time_1, switch_probability, color='red', zorder=1)

    plt.xlabel('Time')
    plt.ylabel('Probability at Node 4')
    plt.title(r'$P_4$ One-Step PST: $\alpha/\sqrt{\beta}=6/\sqrt{7}$')
    plt.savefig("figure.pdf")
    plt.show()
