# Random Clifford circuit builder
# Alternates layers of random 2-qubit Clifford gates with single-qubit measurements
import numpy as np
from .stabilizer import StabilizerState


def random_clifford_circuit(state, p_meas, rng=None):
    """
    Apply one layer of the hybrid random Clifford circuit to state in-place.
    p_meas: measurement probability per qubit per layer.
    rng: numpy Generator instance.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = state.n
    # placeholder: full gate implementations live in stabilizer.py
    pass
