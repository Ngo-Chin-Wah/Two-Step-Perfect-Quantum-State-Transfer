import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths


def generate_p4_hamiltonian(weights):
    """Generates the P4 Hamiltonian based on edge weights."""
    h = np.zeros((4, 4))
    h[0, 1] = h[1, 0] = weights[0]
    h[1, 2] = h[2, 1] = weights[1]
    h[2, 3] = h[3, 2] = weights[2]
    return h.astype(np.complex128)


def evolve_state(hamiltonian, initial_state, timesteps, dt):
    """Evolves the state step by step using Hamiltonian dynamics."""
    evolved_states = []
    state = initial_state
    u = expm(-1j * hamiltonian * dt)  # Time evolution operator for small steps

    for _ in range(timesteps):
        state = np.dot(u, state)
        evolved_states.append(state)

    return np.array(evolved_states)


def calculate_fidelity(hamiltonian, initial_state, target_state, time):
    """Calculates fidelity by evolving the initial state."""
    time_evolution_operator = expm(-1j * hamiltonian * time)
    evolved_state = np.dot(time_evolution_operator, initial_state)
    fidelity = np.abs(np.vdot(target_state, evolved_state)) ** 2
    return fidelity, evolved_state


def detect_sharp_peaks(fidelities, fs, dt, threshold=0.1, prominence=0.05):
    """
    Detects sharp peaks in the Fourier Transform of a signal.

    Parameters:
    - fidelities: Fidelity values over time
    - fs: Sampling frequency
    - dt: Time step
    - threshold: Minimum peak height (relative to the max peak)
    - prominence: Minimum prominence for peak detection

    Returns:
    - peaks: Indices of sharp peaks
    - peak_freqs: Frequencies corresponding to sharp peaks
    """
    # Perform Fourier Transform
    fft_output = fft(fidelities)
    frequencies = fftfreq(len(fidelities), dt)
    magnitudes = np.abs(fft_output[:len(fidelities) // 2])  # Positive frequencies only

    # Normalize magnitudes (relative to max peak)
    magnitudes /= np.max(magnitudes)

    # Detect peaks
    peaks, properties = find_peaks(magnitudes, height=threshold, prominence=prominence)

    # Calculate peak widths to assess sharpness
    results_half = peak_widths(magnitudes, peaks, rel_height=0.5)

    # Get frequencies of detected peaks
    peak_freqs = frequencies[peaks]

    # Plot results
    plt.figure(figsize=(12, 6))

    # Time series fidelity plot
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(fidelities)) * dt, fidelities)
    plt.xlabel("Time (s)")
    plt.ylabel("Fidelity")
    plt.title("Fidelity Over Time")

    # Fourier transform with peak detection
    plt.subplot(1, 2, 2)
    plt.plot(frequencies[:len(fidelities) // 2], magnitudes, label='Magnitude Spectrum')
    plt.scatter(peak_freqs, magnitudes[peaks], color='red', label='Detected Peaks')
    plt.title("Fourier Transform and Peak Detection")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('figure.pdf')
    plt.show()

    # Print sharp peaks and frequencies
    print(f"Detected {len(peaks)} peaks.")
    print(f"Peak frequencies: {np.round(peak_freqs, 3)} Hz")
    print(f"Peak widths (frequency resolution): {np.round(results_half[0], 3)}")

    # Assess if peaks are sharp
    sharp_peaks = sum(results_half[0] < 5)  # Threshold width < 5 bins (adjustable)
    if sharp_peaks > 0:
        print("Sharp peak(s) detected!")
    else:
        print("No sharp peaks detected (peaks are too broad).")

    return peaks, peak_freqs


def periodicity_test_P4(initial_edge_weights, final_edge_weights, run_time_1, run_time_2,
                        timesteps=3000, dt=0.01):
    """
    Periodicity test for P4 Hamiltonian by evolving the state stepwise
    and performing Fourier analysis on the resulting fidelity.
    """

    # Initial state (excitation at node 1)
    initial_state = np.array([1, 0, 0, 0], dtype=np.complex128)[:, np.newaxis]
    target_state = np.array([0, 0, 1, 0], dtype=np.complex128)[:, np.newaxis]

    # Normalize states
    initial_state = initial_state / np.linalg.norm(initial_state)
    target_state = target_state / np.linalg.norm(target_state)

    # Generate Hamiltonians
    H1 = generate_p4_hamiltonian(initial_edge_weights)
    H2 = generate_p4_hamiltonian(final_edge_weights)

    # Time evolution phase 1 (before adjustment)
    _, intermediate_state = calculate_fidelity(H1, initial_state, target_state, run_time_1)

    # Time evolution phase 2 (after adjustment)
    fidelities = []
    states = evolve_state(H2, intermediate_state, timesteps, dt)

    for state in states:
        fidelity = np.abs(np.vdot(target_state, state)) ** 2
        fidelities.append(fidelity)

    # Perform peak detection on Fourier Transform of fidelity
    detect_sharp_peaks(fidelities, timesteps / (dt * timesteps), dt)


initial_edge_weights = [1, 2, 1]  # Initial symmetric edge weights
final_edge_weights = [1, -2, 1]  # Flipped final edge weights
run_time_1 = np.pi / (2 * np.sqrt(2))  # Runtime before adjustment
run_time_2 = np.pi / (2 * np.sqrt(2))  # Runtime after adjustment

periodicity_test_P4(initial_edge_weights, final_edge_weights, run_time_1, run_time_2)