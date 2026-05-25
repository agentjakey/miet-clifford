"""
Unit tests for the stabilizer tableau engine and entropy computation.
"""

import numpy as np
import pytest

from miet_clifford.stabilizer import StabilizerState
from miet_clifford.entropy import (
    entanglement_entropy,
    half_chain_entropy,
    gf2_rank,
    _gf2_rank,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symp_ip(row_i, row_j, n):
    """Symplectic inner product of two tableau rows.
    Returns 0 if the Pauli operators commute, 1 if they anticommute.
    """
    x_i, z_i = row_i[:n].astype(int), row_i[n : 2 * n].astype(int)
    x_j, z_j = row_j[:n].astype(int), row_j[n : 2 * n].astype(int)
    return int((np.dot(x_i, z_j) + np.dot(z_i, x_j)) % 2)


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
        row = s.tableau[n + i]
        expected = np.zeros(2 * n + 1, dtype=np.int8)
        expected[n + i] = 1
        np.testing.assert_array_equal(
            row, expected, err_msg=f"Stabilizer {i} wrong: {row}"
        )


def test_initial_destabilizers():
    s = StabilizerState(4)
    n = 4
    for i in range(n):
        row = s.tableau[i]
        expected = np.zeros(2 * n + 1, dtype=np.int8)
        expected[i] = 1
        np.testing.assert_array_equal(
            row, expected, err_msg=f"Destabilizer {i} wrong: {row}"
        )


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def test_product_state_entropy_zero():
    s = StabilizerState(4)
    assert entanglement_entropy(s, [0, 1]) == 0.0
    assert entanglement_entropy(s, [0]) == 0.0
    assert half_chain_entropy(s) == 0.0


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
    from miet_clifford.circuit import run_circuit

    s = run_circuit(n_qubits=10, p_meas=0.0, n_steps=30)
    e = half_chain_entropy(s)
    assert e > 1.5, f"Volume-law entropy at p=0 too low: {e}"


def test_circuit_area_law():
    """p=0.5, large depth -> half-chain entropy should be low."""
    from miet_clifford.circuit import run_circuit

    vals = [half_chain_entropy(run_circuit(10, 0.5, 50)) for _ in range(10)]
    assert np.mean(vals) < 2.0, f"Area-law entropy at p=0.5 too high: {np.mean(vals)}"


def test_volume_exceeds_area():
    """Volume-law mean entropy must exceed area-law mean entropy."""
    from miet_clifford.circuit import run_circuit

    n = 12
    vol = np.mean([half_chain_entropy(run_circuit(n, 0.0, 20)) for _ in range(5)])
    area = np.mean([half_chain_entropy(run_circuit(n, 0.5, 50)) for _ in range(5)])
    assert vol > area, f"Volume ({vol:.2f}) not > area ({area:.2f})"


# ---------------------------------------------------------------------------
# Single-qubit Clifford conjugation
# ---------------------------------------------------------------------------


def test_h_conjugates_x_to_z():
    """HXH† = Z: state |+> (stabilizer X) transforms to |0> (stabilizer Z)."""
    s = StabilizerState(1)
    s.apply_h(0)  # |0> -> |+>,  stabilizer becomes X
    s.apply_h(0)  # HXH† = Z,    stabilizer becomes Z
    row = s.tab[1]  # stabilizer row
    assert (
        row[0] == 0 and row[1] == 1 and row[2] == 0
    ), f"HXH† != Z: tab row = {list(row)}, expected [0,1,0]"


def test_h_conjugates_z_to_x():
    """HZH† = X: state |0> (stabilizer Z) transforms to |+> (stabilizer X)."""
    s = StabilizerState(1)
    s.apply_h(0)  # HZH† = X
    row = s.tab[1]
    assert (
        row[0] == 1 and row[1] == 0 and row[2] == 0
    ), f"HZH† != X: tab row = {list(row)}, expected [1,0,0]"


def test_h_conjugates_y_to_minus_y():
    """HYH† = -Y: verify phase bit is set and X,Z bits match Y."""
    s = StabilizerState(1)
    s.apply_h(0)  # stabilizer X
    s.apply_s(0)  # SXS† = Y -> stabilizer Y: [1,1,0]
    assert list(s.tab[1]) == [1, 1, 0], f"Pre-H stabilizer not Y: {list(s.tab[1])}"
    s.apply_h(0)  # HYH† = -Y: x=1, z=1, phase=1
    row = s.tab[1]
    assert (
        row[0] == 1 and row[1] == 1 and row[2] == 1
    ), f"HYH† != -Y: tab row = {list(row)}, expected [1,1,1]"


def test_s_conjugates_x_to_y():
    """SXS† = Y: state |+> (stabilizer X) becomes stabilizer Y."""
    s = StabilizerState(1)
    s.apply_h(0)  # stabilizer X: tab[1] = [1,0,0]
    s.apply_s(0)  # SXS† = Y
    row = s.tab[1]
    assert (
        row[0] == 1 and row[1] == 1 and row[2] == 0
    ), f"SXS† != Y: tab row = {list(row)}, expected [1,1,0]"


def test_s_conjugates_z_to_z():
    """SZS† = Z: state |0> (stabilizer Z) stabilizer is unchanged."""
    s = StabilizerState(1)
    s.apply_s(0)  # SZS† = Z
    row = s.tab[1]
    assert (
        row[0] == 0 and row[1] == 1 and row[2] == 0
    ), f"SZS† != Z: tab row = {list(row)}, expected [0,1,0]"


def test_s_conjugates_y_to_minus_x():
    """SYS† = -X: phase bit set, x=1, z=0."""
    s = StabilizerState(1)
    s.apply_h(0)
    s.apply_s(0)  # stabilizer Y: [1,1,0]
    s.apply_s(0)  # SYS† = -X: [1,0,1]
    row = s.tab[1]
    assert (
        row[0] == 1 and row[1] == 0 and row[2] == 1
    ), f"SYS† != -X: tab row = {list(row)}, expected [1,0,1]"


# ---------------------------------------------------------------------------
# CNOT conjugation
# ---------------------------------------------------------------------------


def test_cnot_conjugation_tableau():
    """CNOT maps X_c->X_c X_t, X_t->X_t, Z_c->Z_c, Z_t->Z_c Z_t.

    Verified directly on the initial 2-qubit tableau where destabilizers are
    X_0, X_1 and stabilizers are Z_0, Z_1.
    Column layout (n=2): [X_0, X_1, Z_0, Z_1, phase].
    """
    s = StabilizerState(2)
    s.apply_cnot(0, 1)
    tab = s.tab
    assert list(tab[0]) == [
        1,
        1,
        0,
        0,
        0,
    ], f"X_c (D_0) -> X_c X_t failed: {list(tab[0])}"
    assert list(tab[1]) == [0, 1, 0, 0, 0], f"X_t (D_1) -> X_t failed: {list(tab[1])}"
    assert list(tab[2]) == [0, 0, 1, 0, 0], f"Z_c (S_0) -> Z_c failed: {list(tab[2])}"
    assert list(tab[3]) == [
        0,
        0,
        1,
        1,
        0,
    ], f"Z_t (S_1) -> Z_c Z_t failed: {list(tab[3])}"


# ---------------------------------------------------------------------------
# Bell state entropy
# ---------------------------------------------------------------------------


def test_bell_state_entropy_both_qubits():
    """S(A) = 1 ebit for either single qubit of |Phi+>."""
    s = StabilizerState(2)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    e0 = entanglement_entropy(s, [0])
    e1 = entanglement_entropy(s, [1])
    assert e0 == 1.0, f"|Phi+>: S(qubit 0) = {e0}, expected 1.0"
    assert e1 == 1.0, f"|Phi+>: S(qubit 1) = {e1}, expected 1.0"


# ---------------------------------------------------------------------------
# GHZ state entropies
# ---------------------------------------------------------------------------


def test_ghz3_single_qubit_entropy():
    """GHZ(3): every single-qubit reduced state is maximally mixed -> S=1."""
    s = StabilizerState(3)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    s.apply_cnot(0, 2)
    for q in range(3):
        e = entanglement_entropy(s, [q])
        assert e == 1.0, f"GHZ(3) S({{qubit {q}}}) = {e}, expected 1.0"


def test_ghz3_half_chain_entropy():
    """GHZ(3): half-chain entropy (S of qubit 0 alone, n//2=1) = 1."""
    s = StabilizerState(3)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    s.apply_cnot(0, 2)
    assert half_chain_entropy(s) == 1.0


def test_ghz4_single_qubit_entropy():
    """GHZ(4): every single-qubit reduced state is maximally mixed -> S=1."""
    s = StabilizerState(4)
    s.apply_h(0)
    for i in range(3):
        s.apply_cnot(0, i + 1)
    for q in range(4):
        e = entanglement_entropy(s, [q])
        assert e == 1.0, f"GHZ(4) S({{qubit {q}}}) = {e}, expected 1.0"


# ---------------------------------------------------------------------------
# Stabilizer commutation invariant
# ---------------------------------------------------------------------------


def test_stabilizer_commutation_after_random_circuit():
    """All stabilizer generators must mutually commute after any Clifford sequence."""
    rng = np.random.default_rng(42)
    n = 5
    s = StabilizerState(n)
    for _ in range(40):
        op = int(rng.integers(4))
        q = int(rng.integers(n))
        if op == 0:
            s.apply_h(q)
        elif op == 1:
            s.apply_s(q)
        elif op == 2:
            q2 = int((q + 1 + rng.integers(n - 1)) % n)
            s.apply_cnot(q, q2)
        else:
            s.measure(q)
    stabs = s.tab[n : 2 * n]
    for i in range(n):
        for j in range(i + 1, n):
            ip = _symp_ip(stabs[i], stabs[j], n)
            assert ip == 0, f"Stabilizers {i} and {j} anticommute after random circuit"


# ---------------------------------------------------------------------------
# Canonical symplectic pairing (destabilizer/stabilizer)
# ---------------------------------------------------------------------------


def test_symplectic_pairing_after_clifford_gates():
    """d_i must anticommute with s_i and commute with s_j (j != i).

    This canonical pairing is maintained by gate operations alone (not checked
    after measurement, which rearranges the pairing).
    """
    rng = np.random.default_rng(17)
    n = 4
    s = StabilizerState(n)
    for _ in range(30):
        op = int(rng.integers(3))
        q = int(rng.integers(n))
        if op == 0:
            s.apply_h(q)
        elif op == 1:
            s.apply_s(q)
        else:
            q2 = int((q + 1 + rng.integers(n - 1)) % n)
            s.apply_cnot(q, q2)
    destabs = s.tab[:n]
    stabs = s.tab[n : 2 * n]
    for i in range(n):
        for j in range(n):
            ip = _symp_ip(destabs[i], stabs[j], n)
            expected = 1 if i == j else 0
            assert (
                ip == expected
            ), f"d_{i} vs s_{j}: expected symplectic product={expected}, got {ip}"


# ---------------------------------------------------------------------------
# Measurement repeatability
# ---------------------------------------------------------------------------


def test_measurement_repeatability_after_random_clifford():
    """Measuring the same qubit twice in immediate succession is deterministic."""
    rng = np.random.default_rng(5)
    n = 3
    for trial in range(30):
        s = StabilizerState(n)
        for _ in range(8):
            op = int(rng.integers(3))
            q = int(rng.integers(n))
            if op == 0:
                s.apply_h(q)
            elif op == 1:
                s.apply_s(q)
            else:
                q2 = int((q + 1) % n)
                s.apply_cnot(q, q2)
        q_meas = int(rng.integers(n))
        o1 = s.measure(q_meas)
        o2 = s.measure(q_meas)
        assert (
            o1 == o2
        ), f"Trial {trial}: measuring qubit {q_meas} twice gave {o1} then {o2}"


# ---------------------------------------------------------------------------
# Entropy is always a non-negative integer for stabilizer states
# ---------------------------------------------------------------------------


def test_entropy_is_nonneg_integer_after_random_circuit():
    """Stabilizer entanglement entropy must be a non-negative integer in {0,...,|A|}.

    Any real-valued (non-integer) result indicates a bug in the GF(2) rank
    computation (e.g., floating-point rank instead of binary rank).
    """
    from miet_clifford.circuit import run_circuit

    rng = np.random.default_rng(13)
    n = 8
    for trial in range(20):
        s = run_circuit(n, p_meas=0.1, n_steps=30, rng=rng)
        for size in [1, 2, n // 2, n - 1]:
            A = list(range(size))
            e = entanglement_entropy(s, A)
            assert e == int(e), (
                f"Trial {trial}: entropy for |A|={size} is {e} (not integer). "
                "GF(2) rank computation may be using floating-point arithmetic."
            )
            assert 0 <= e <= min(size, n - size), (
                f"Trial {trial}: entropy {e} out of range [0, {min(size, n-size)}] "
                f"for |A|={size}, n={n}."
            )


# ---------------------------------------------------------------------------
# GF(2) rank vs real rank: constructed counter-example
# ---------------------------------------------------------------------------


def test_gf2_rank_differs_from_real_rank():
    """Show that numpy real-valued rank gives WRONG answer for a GF(2) matrix.

    The matrix [[1,1],[1,1]] has real rank 1 (rows are proportional) AND
    GF(2) rank 1 (rows sum to [0,0] mod 2, so they are linearly dependent mod 2).

    A matrix where real rank != GF(2) rank:
        [[1,1,0],[0,1,1],[1,0,1]]
    Over the reals: rank 3 (all rows independent over R).
    Over GF(2): rows sum to [0,0,0] mod 2 -> rank 2.

    This test documents why gf2_rank (XOR elimination) is required and
    numpy.linalg.matrix_rank cannot be substituted.
    """
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int8)
    gf2 = gf2_rank(M)
    real = int(np.linalg.matrix_rank(M.astype(float)))
    assert gf2 == 2, f"GF(2) rank should be 2, got {gf2}"
    assert real == 3, f"Real rank should be 3, got {real}"
    assert gf2 != real, "GF(2) rank and real rank must differ on this matrix"


# ---------------------------------------------------------------------------
# Entanglement profile: S(A) shape in volume-law and area-law circuits
# ---------------------------------------------------------------------------


def test_entropy_profile_volume_law():
    """At p=0, S(A) grows with |A|: S(L/2) > S(L/4) > S(1)."""
    from miet_clifford.circuit import run_circuit

    rng = np.random.default_rng(2025)
    n = 16
    s = run_circuit(n, p_meas=0.0, n_steps=40, rng=rng)
    e1 = entanglement_entropy(s, [0])
    e4 = entanglement_entropy(s, list(range(n // 4)))
    e8 = entanglement_entropy(s, list(range(n // 2)))
    assert e8 > e4, f"Volume law: S(L/2)={e8} should exceed S(L/4)={e4}"
    assert e4 > e1, f"Volume law: S(L/4)={e4} should exceed S(1)={e1}"


def test_entropy_profile_area_law():
    """At p=0.5, S(A) is small and nearly flat across subsystem sizes."""
    from miet_clifford.circuit import run_circuit

    rng = np.random.default_rng(777)
    n = 16
    vals = []
    for _ in range(10):
        s = run_circuit(n, p_meas=0.5, n_steps=50, rng=rng)
        e_small = entanglement_entropy(s, list(range(2)))
        e_half = entanglement_entropy(s, list(range(n // 2)))
        vals.append((e_small, e_half))

    mean_small = np.mean([v[0] for v in vals])
    mean_half = np.mean([v[1] for v in vals])
    assert mean_half < 3.0, f"Area law: mean S(L/2)={mean_half:.2f} too large at p=0.5"
    assert mean_half - mean_small < 3.0, (
        f"Area law: S(L/2)-S(2)={mean_half-mean_small:.2f} too large;"
        " entropy should be approximately flat"
    )
