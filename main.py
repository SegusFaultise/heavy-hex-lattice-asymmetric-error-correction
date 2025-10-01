import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pyvis.network import Network
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2
from qiskit import (QuantumCircuit, QuantumRegister,
                    ClassicalRegister, transpile)


if not os.path.exists('plots'):
    os.makedirs('plots')

backend = FakeMontrealV2()
config = backend.configuration()
coupling_map = config.coupling_map

NUM_DATA_QUBITS = config.n_qubits
print(f"Loaded '{backend.name}' backend with {NUM_DATA_QUBITS} qubits.")

VERTEX_STABILISERS = {
    "v0_Z": {"qubits": [2, 5, 8], "type": "Z"},
    "v1_Z": {"qubits": [18, 21, 24], "type": "Z"},
}
FACE_STABILISERS = {
    "f0_X": {"qubits": [7, 8, 10, 11, 12, 13, 14], "type": "X"},
}
ALL_STABILISERS = {**VERTEX_STABILISERS, **FACE_STABILISERS}


def create_error_model(p_error: float, error_type: str) -> NoiseModel:
    """Creates a noise model for a specific type of Pauli error."""
    if error_type == 'bit_flip':
        error_1 = pauli_error([('X', p_error), ('I', 1 - p_error)])
    elif error_type == 'phase_flip':
        error_1 = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    else:
        raise ValueError("Invalid error type.")

    error_2 = error_1.tensor(error_1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz'])

    return noise_model


def build_stabilizer_measurement_circuit(num_qubits, stabiliser_qubits, stabiliser_type, insert_error=False):
    """Builds a circuit to measure a single stabiliser."""
    q_data = QuantumRegister(num_qubits, name='data')
    q_ancilla = QuantumRegister(1, name='ancilla')
    c_result = ClassicalRegister(1, name='result')
    circuit = QuantumCircuit(q_data, q_ancilla, c_result)

    if stabiliser_type == 'X':
        for i in stabiliser_qubits:
            circuit.h(q_data[i])

    circuit.barrier()

    if insert_error:
        error_qubit = stabiliser_qubits[0]

        if stabiliser_type == 'Z':
            circuit.x(q_data[error_qubit])
        elif stabiliser_type == 'X':
            circuit.z(q_data[error_qubit])

        circuit.barrier()

    circuit.h(q_ancilla[0])

    if stabiliser_type == 'Z':
        for i in stabiliser_qubits:
            circuit.cz(q_data[i], q_ancilla[0])

    elif stabiliser_type == 'X':
        for i in stabiliser_qubits:
            circuit.cx(q_ancilla[0], q_data[i])

    circuit.h(q_ancilla[0])
    circuit.measure(q_ancilla[0], c_result[0])

    return circuit


def plot_lattice_with_stabilisers(coupling_map, stabilisers):
    """Plots the full hardware lattice and highlights all specified stabilisers."""
    print("\n--- Generating Lattice Overview Plot ---")
    positions = {
        0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1), 4: (2, 1), 5: (2, -1),
        6: (3, 2), 7: (3, 1), 8: (3, -1), 9: (3, -2), 10: (4, 1), 11: (4, -1),
        12: (5, 1), 13: (5, 0), 14: (5, -1), 15: (6, 1), 16: (6, -1),
        17: (7, 2), 18: (7, 1), 19: (7, -1), 20: (7, -2), 21: (8, 1),
        22: (8, -1), 23: (9, 1), 24: (9, 0), 25: (9, -1), 26: (10, -1)
    }

    G = nx.Graph(coupling_map)
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    nx.draw(G, pos=positions, with_labels=True, node_color='lightgray',
            node_size=400, font_size=8, edge_color='#e0e0e0', ax=ax)

    for name, s in stabilisers.items():
        qubits = s['qubits']
        color = 'blue' if s['type'] == 'Z' else 'red'

        nx.draw_networkx_nodes(
            G, pos=positions, nodelist=qubits, node_color=color, node_size=500, ax=ax)

        label_pos = np.mean([positions[q] for q in qubits], axis=0)
        ax.text(label_pos[0], label_pos[1] + 0.3, name,
                ha='center', fontsize=9, fontweight='bold', color=color)

    nx.draw_networkx_labels(G, pos=positions, font_size=8)

    filename = "plots/lattice_overview.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved lattice overview to {filename}")


def plot_interactive_lattice(coupling_map, stabilisers, num_qubits, filename="plots/interactive_lattice.html"):
    """Generates an interactive HTML plot with detailed hover information."""
    print("\n--- Generating Interactive Lattice Plot ---")

    G = nx.Graph(coupling_map)
    degrees = {node: G.degree(node) for node in G.nodes()}
    qubit_to_stabilisers = {i: [] for i in range(num_qubits)}

    for name, s in stabilisers.items():
        for qubit in s['qubits']:
            qubit_to_stabilisers[qubit].append(name)

    net = Network(height="800px", width="100%", notebook=True)

    for i in range(num_qubits):
        title = f"Qubit: {i}\nDegree: {degrees.get(i, 0)}\nStabilisers: {
            ', '.join(qubit_to_stabilisers[i]) or 'None'}"

        color = 'lightgray'
        if 'f0_X' in qubit_to_stabilisers[i]:
            color = '#c0392b'
        elif qubit_to_stabilisers[i]:
            color = '#2980b9'

        net.add_node(i, label=str(i), title=title, color=color)

    net.add_edges(coupling_map)
    net.save_graph(filename)

    print(f"Saved interactive lattice to {filename}")


def plot_circuit_diagrams(num_qubits, stabilisers):
    """Generates and saves diagrams of the measurement circuits."""
    print("\n--- Generating Circuit Diagrams ---")
    for name, s in stabilisers.items():
        circuit = build_stabilizer_measurement_circuit(
            num_qubits, s['qubits'], s['type'], insert_error=True)

        fig = circuit.draw('mpl', style='iqx', fold=-1)
        filename = f"plots/circuit_diagram_{name}_with_error.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)

        print(f"Saved circuit diagram for '{name}' to {filename}")


def plot_error_thresholds(p_phys_range, results, stabiliser_name):
    """Generates the final log-log plot of failure rates vs. physical error rates."""
    print("\n--- Generating Error Threshold Plot ---")
    plt.figure(figsize=(10, 6))

    plt.plot(p_phys_range, results['bit_flip'],
             'o-', label='Bit-Flip (X) Noise')
    plt.plot(p_phys_range, results['phase_flip'],
             's-', label='Phase-Flip (Z) Noise')
    plt.plot(p_phys_range, p_phys_range, 'k--', alpha=0.5,
             label='Reference: p_failure = p_physical')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate ($p_{phys}$)', fontsize=12)
    plt.ylabel('Stabiliser Failure Rate ($p_{fail}$)', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    filename = f"plots/error_threshold_plot_{stabiliser_name}.png"

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved error threshold plot for '{stabiliser_name}' to {filename}")


if __name__ == "__main__":
    plot_lattice_with_stabilisers(coupling_map, ALL_STABILISERS)
    plot_circuit_diagrams(NUM_DATA_QUBITS, ALL_STABILISERS)

    stabiliser_to_test = FACE_STABILISERS['f0_X']
    stabiliser_name = 'f0_X'

    p_phys_range = np.logspace(-3.5, -1.5, 10)
    shots_per_point = 8192

    threshold_results = {}
    error_types_to_test = ['bit_flip', 'phase_flip']

    for error_type in error_types_to_test:
        print(
            f"\n--- Running threshold analysis for '{stabiliser_name}' with {error_type} noise ---")
        failure_rates = []

        for p_phys in p_phys_range:
            print(f"  Testing physical error rate: {p_phys:.5f}")

            noise_model = create_error_model(p_phys, error_type)
            noisy_simulator = AerSimulator(noise_model=noise_model)

            circuit = build_stabilizer_measurement_circuit(
                NUM_DATA_QUBITS, stabiliser_to_test['qubits'], stabiliser_to_test['type'], insert_error=False
            )

            transpiled_circuit = transpile(circuit, noisy_simulator)

            result = noisy_simulator.run(
                transpiled_circuit, shots=shots_per_point).result()

            counts = result.get_counts(0)

            failure_rate = counts.get('1', 0) / shots_per_point
            failure_rates.append(failure_rate)

        threshold_results[error_type] = failure_rates

    plot_error_thresholds(p_phys_range, threshold_results, stabiliser_name)
    plot_interactive_lattice(coupling_map, ALL_STABILISERS, NUM_DATA_QUBITS)

    print("\n\nAll simulations and plotting complete.")
