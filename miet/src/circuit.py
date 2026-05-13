"""
Random Clifford circuit primitives and brickwork hybrid circuit for MIPT simulations.
"""
import numpy as np
from .stabilizer import StabilizerState


SINGLE_QUBIT_CLIFFORDS = [
    (),
    ('H',),
    ('S',),
    ('S', 'S'),
    ('S', 'S', 'S'),
    ('H', 'S'),
    ('H', 'S', 'S'),
    ('H', 'S', 'S', 'S'),
    ('S', 'H'),
    ('S', 'S', 'H'),
    ('S', 'S', 'S', 'H'),
    ('H', 'S', 'H'),
    ('S', 'H', 'S'),
    ('S', 'S', 'H', 'S'),
    ('S', 'S', 'S', 'H', 'S'),
    ('H', 'S', 'S', 'H'),
    ('S', 'H', 'S', 'S'),
    ('S', 'S', 'H', 'S', 'S'),
    ('H', 'S', 'H', 'S'),
    ('H', 'S', 'H', 'S', 'S'),
    ('H', 'S', 'H', 'S', 'S', 'S'),
    ('S', 'H', 'S', 'H'),
    ('S', 'H', 'S', 'H', 'S'),
    ('S', 'H', 'S', 'H', 'S', 'S'),
]
assert len(SINGLE_QUBIT_CLIFFORDS) == 24


def apply_single_qubit_clifford(state: StabilizerState, qubit: int, gates: tuple):
    for g in gates:
        if g == 'H':
            state.apply_h(qubit)
        else:
            state.apply_s(qubit)


def random_single_qubit_clifford(state: StabilizerState, qubit: int):
    apply_single_qubit_clifford(state, qubit,
                                SINGLE_QUBIT_CLIFFORDS[np.random.randint(24)])


def random_two_qubit_clifford(state: StabilizerState, q0: int, q1: int):
    """Random 2-qubit Clifford via decomposition into 1Q Cliffords and CNOTs.

    Samples approximately uniformly from the 2-qubit Clifford group (11,520
    elements). Approximate uniformity is sufficient for MIPT physics; see
    Li et al. (2019).
    """
    random_single_qubit_clifford(state, q0)
    random_single_qubit_clifford(state, q1)
    state.apply_cnot(q0, q1)
    random_single_qubit_clifford(state, q0)
    random_single_qubit_clifford(state, q1)
    state.apply_cnot(q1, q0)
    random_single_qubit_clifford(state, q0)
    random_single_qubit_clifford(state, q1)


def brickwork_layer(state: StabilizerState, layer_index: int):
    """Brickwork pattern of random 2-qubit Cliffords.

    Even layers: pairs (0,1),(2,3),(4,5),...
    Odd  layers: pairs (1,2),(3,4),(5,6),...
    Boundary qubits that cannot be paired get a random single-qubit Clifford.
    """
    n = state.n
    start = layer_index % 2
    paired = set()
    for i in range(start, n - 1, 2):
        random_two_qubit_clifford(state, i, i + 1)
        paired.add(i)
        paired.add(i + 1)
    for i in range(n):
        if i not in paired:
            random_single_qubit_clifford(state, i)


def measurement_layer(state: StabilizerState, p: float):
    """Measure each qubit independently with probability p. Outcomes are discarded."""
    for i in range(state.n):
        if np.random.rand() < p:
            state.measure(i)


def run_circuit(n_qubits: int, p_meas: float, n_steps: int,
                warmup: int = None) -> StabilizerState:
    """Run hybrid random Clifford + measurement circuit to steady state.

    Warmup uses the same p_meas rate as the main run — essential for reaching
    the correct steady state of the hybrid Markov chain at rate p_meas.

    Args:
        n_qubits: system size
        p_meas:   single-site measurement probability per layer (0 to 1)
        n_steps:  brickwork + measurement cycles after warmup
        warmup:   thermalization cycles (default 3 * n_qubits)
    Returns:
        StabilizerState in steady state at rate p_meas.
    """
    if warmup is None:
        warmup = 3 * n_qubits

    state = StabilizerState(n_qubits)

    for t in range(warmup):
        brickwork_layer(state, t)
        measurement_layer(state, p_meas)

    for t in range(n_steps):
        brickwork_layer(state, warmup + t)
        measurement_layer(state, p_meas)

    return state
