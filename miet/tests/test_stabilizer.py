"""
Unit tests for the stabilizer tableau engine and entropy computation.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.stabilizer import StabilizerState
from src.entropy import entanglement_entropy, half_chain_entropy, gf2_rank, _gf2_rank


# ---------------------------------------------------------------------------
# Tableau initialisation
# ---------------------------------------------------------------------------

def test_initial_state_shape():
    s = StabilizerState(4)
    assert s.tableau.shape == (8, 9)
    assert s.tableau.dtype == np.int8


def test_initial_stabilizers():
    s = StabilizerState(4)
    n = 4
    for i in range(n):
        row = s.tableau[n+i]
        expected = np.zeros(2*n+1, dtype=np.int8)
        expected[n+i] = 1
        np.testing.assert_array_equal(row, expected,
            err_msg=f"Stabilizer {i} wrong: {row}")


def test_initial_destabilizers():
    s = StabilizerState(4)
    n = 4
    for i in range(n):
        row = s.tableau[i]
        expected = np.zeros(2*n+1, dtype=np.int8)
        expected[i] = 1
        np.testing.assert_array_equal(row, expected,
            err_msg=f"Destabilizer {i} wrong: {row}")


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_product_state_entropy_zero():
    s = StabilizerState(4)
    assert entanglement_entropy(s, [0, 1]) == 0.0
    assert entanglement_entropy(s, [0])    == 0.0
    assert half_chain_entropy(s)           == 0.0


def test_bell_state_entropy():
    """PRIMARY CORRECTNESS TEST for the entropy formula.
    |Phi+> has S(A=qubit0) = 1.0 ebit exactly.
    If this fails: check that entropy uses complement B columns, not A columns.
    S(A) = |A| - n + rank(M_B) with B = complement of A.
    """
    s = StabilizerState(2)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    result = half_chain_entropy(s)
    assert result == 1.0, (
        f"Bell state entropy = {result}, expected 1.0. "
        "Entropy formula is wrong -- verify S(A) = |A| - n + rank(M_B) using "
        "complement B's columns, not A's columns."
    )


def test_ghz_entropy():
    """4-qubit GHZ: S({0,1}) = 1.0 ebit (not 2.0 -- GHZ has bipartite entropy 1)."""
    s = StabilizerState(4)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    s.apply_cnot(0, 2)
    s.apply_cnot(0, 3)
    e = entanglement_entropy(s, [0, 1])
    assert e == 1.0, f"GHZ half-chain entropy = {e}, expected 1.0"


def test_measurement_collapses_entropy():
    """After measuring all qubits of a 2-qubit |++> state, entropy is 0."""
    for _ in range(20):
        s = StabilizerState(2)
        s.apply_h(0)
        s.apply_h(1)
        s.measure(0)
        s.measure(1)
        assert half_chain_entropy(s) == 0.0


# ---------------------------------------------------------------------------
# Gate correctness
# ---------------------------------------------------------------------------

def test_bell_state_correlated_outcomes():
    """Measuring qubit 0 then qubit 1 of |Phi+> must always agree."""
    for _ in range(100):
        s = StabilizerState(2)
        s.apply_h(0)
        s.apply_cnot(0, 1)
        o0 = s.measure(0)
        o1 = s.measure(1)
        assert o0 == o1, f"Bell state disagreement: {o0} vs {o1}"


def test_hadamard_random_outcomes():
    """H|0> = |+>; Z measurement must be random (roughly 50/50)."""
    count_1 = 0
    for _ in range(400):
        s = StabilizerState(1)
        s.apply_h(0)
        count_1 += s.measure(0)
    assert 140 <= count_1 <= 260, (
        f"H|0> measurement bias: {count_1}/400 ones, expected ~200. "
        "Check H gate phase update uses x_old & z_old before swap."
    )


def test_s_gate_correct_phase():
    """S^2|0> = Z|0> = |0> (Z has eigenvalue +1 for |0>). Outcome must be 0."""
    for _ in range(20):
        s = StabilizerState(1)
        s.apply_s(0)
        s.apply_s(0)
        o = s.measure(0)
        assert o == 0, (
            f"S^2|0> measurement = {o}, expected 0. "
            "Check S gate: phase update must use OLD z before z is modified."
        )


# ---------------------------------------------------------------------------
# GF(2) rank
# ---------------------------------------------------------------------------

def test_gf2_rank_known_cases():
    assert gf2_rank(np.eye(3, dtype=np.int8)) == 3
    assert gf2_rank(np.zeros((3, 3), dtype=np.int8)) == 0
    assert gf2_rank(np.array([[1, 1], [1, 1]], dtype=np.int8)) == 1
    # Rows sum to zero mod 2 -> rank 2
    B = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int8)
    assert gf2_rank(B) == 2


# ---------------------------------------------------------------------------
# Measurement -- deterministic Case 2
# ---------------------------------------------------------------------------

def test_deterministic_measurement_case2():
    """
    Explicit test for the Case 2 (deterministic) measurement path.

    |0> has stabilizer +Z. Measuring Z must give outcome 0 deterministically.
    This exercises the Case 2 branch: no stabilizer has x[*,a]=1, so the
    outcome is computed via the destabilizer/stabilizer product accumulation.

    Critical checks:
      - scratch must start as +I (all zeros), NOT as Z_a.
      - loop must check DESTABILIZER rows (0..n-1), not stabilizer rows.
    If either is wrong, this test will intermittently return 1 instead of 0.
    """
    for _ in range(50):
        s = StabilizerState(3)
        # All three qubits start in |0>; stabilizers are Z_0, Z_1, Z_2.
        # Measuring Z on qubit 0 is always deterministically 0.
        o = s.measure(0)
        assert o == 0, (
            f"Measuring Z on |0> gave {o}, expected 0. "
            "Case 2 bug: check scratch init (must be zeros, not Z_a) and "
            "loop condition (must check destabilizer rows, not stabilizer rows)."
        )


def test_deterministic_measurement_outcome_one():
    """
    Case 2 must also correctly return outcome 1 when appropriate.
    |1> has stabilizer -Z (phase=1). Measuring Z must give outcome 1.
    Prepare |1> = X|0>, then measure Z.
    """
    for _ in range(50):
        s2 = StabilizerState(1)
        # Apply X gate = H S^2 H to qubit 0
        s2.apply_h(0)
        s2.apply_s(0)
        s2.apply_s(0)
        s2.apply_h(0)
        # s2 represents |1>; stabilizer is -Z (phase=1).
        # Measuring Z should give deterministic outcome 1.
        o = s2.measure(0)
        assert o == 1, (
            f"Measuring Z on |1> gave {o}, expected 1. "
            "Case 2 deterministic outcome=1 path is broken."
        )


# ---------------------------------------------------------------------------
# Circuit-level physics (MIPT phase separation)
# ---------------------------------------------------------------------------

def test_circuit_volume_law():
    """p=0, large depth -> half-chain entropy should be well above 0."""
    from src.circuit import run_circuit
    s = run_circuit(n_qubits=10, p_meas=0.0, n_steps=30)
    e = half_chain_entropy(s)
    assert e > 1.5, f"Volume-law entropy at p=0 too low: {e}"


def test_circuit_area_law():
    """p=0.5, large depth -> half-chain entropy should be low."""
    from src.circuit import run_circuit
    vals = [half_chain_entropy(run_circuit(10, 0.5, 50)) for _ in range(10)]
    assert np.mean(vals) < 2.0, f"Area-law entropy at p=0.5 too high: {np.mean(vals)}"


def test_volume_exceeds_area():
    """Volume-law mean entropy must exceed area-law mean entropy."""
    from src.circuit import run_circuit
    n = 12
    vol  = np.mean([half_chain_entropy(run_circuit(n, 0.0, 20)) for _ in range(5)])
    area = np.mean([half_chain_entropy(run_circuit(n, 0.5, 50)) for _ in range(5)])
    assert vol > area, f"Volume ({vol:.2f}) not > area ({area:.2f})"
