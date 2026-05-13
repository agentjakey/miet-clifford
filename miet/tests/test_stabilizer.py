"""
Unit tests for the stabilizer tableau engine and entropy computation.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.stabilizer import StabilizerState
from src.entropy import entanglement_entropy, _gf2_rank


class TestStabilizerState:
    def test_init_shape(self):
        n = 4
        s = StabilizerState(n)
        assert s.tab.shape == (2 * n, 2 * n + 1)

    def test_init_destabilizers(self):
        n = 4
        s = StabilizerState(n)
        for i in range(n):
            assert s.tab[i, i] == 1, f"destabilizer row {i} should have X_{i} set"

    def test_init_stabilizers(self):
        n = 4
        s = StabilizerState(n)
        for i in range(n):
            assert s.tab[n + i, n + i] == 1, f"stabilizer row {i} should have Z_{i} set"

    def test_init_all_phases_zero(self):
        n = 6
        s = StabilizerState(n)
        assert np.all(s.tab[:, -1] == 0), "all phases should be 0 for |0> state"

    def test_copy_independence(self):
        s = StabilizerState(4)
        t = s.copy()
        t.tab[0, 0] ^= 1
        assert s.tab[0, 0] != t.tab[0, 0], "copy should be independent"

    def test_deterministic_outcome_zero(self):
        """Measuring Z on |0> is deterministic outcome 0 (Case 2 path)."""
        for _ in range(50):
            s = StabilizerState(3)
            o = s.measure(0)
            assert o == 0, (
                f"Z measurement on |0> returned {o}, expected 0. "
                "Case 2 bug: scratch must start as zeros (not Z_a), "
                "and loop must check destabilizer rows (not stabilizer rows)."
            )

    def test_deterministic_outcome_one(self):
        """Measuring Z on |1> is deterministic outcome 1 (Case 2 path, phase=1)."""
        for _ in range(50):
            s = StabilizerState(1)
            # Prepare |1> via X = H S^2 H
            s.apply_h(0)
            s.apply_s(0)
            s.apply_s(0)
            s.apply_h(0)
            o = s.measure(0)
            assert o == 1, (
                f"Z measurement on |1> returned {o}, expected 1. "
                "Case 2 broken: loop never runs so phase stays 0."
            )


class TestGF2Rank:
    def test_identity(self):
        M = np.eye(4, dtype=np.uint8)
        assert _gf2_rank(M) == 4

    def test_zero_matrix(self):
        M = np.zeros((3, 3), dtype=np.uint8)
        assert _gf2_rank(M) == 0

    def test_rank_deficient(self):
        M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
        assert _gf2_rank(M) == 2  # rows sum to 0 over GF(2)

    def test_single_row(self):
        M = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        assert _gf2_rank(M) == 1


class TestEntanglementEntropy:
    def test_product_state_zero_entropy(self):
        # |0>^n has zero entanglement across any bipartition
        n = 6
        s = StabilizerState(n)
        for size in range(1, n):
            S = entanglement_entropy(s, list(range(size)))
            assert S == 0, f"product state entropy should be 0, got {S} for size {size}"

    def test_entropy_nonnegative(self):
        n = 4
        s = StabilizerState(n)
        S = entanglement_entropy(s, [0, 1])
        assert S >= 0

    def test_entropy_bounded_by_subsystem_size(self):
        n = 8
        s = StabilizerState(n)
        size = 4
        S = entanglement_entropy(s, list(range(size)))
        assert S <= size, "entropy bounded by min(|A|, |A^c|)"
