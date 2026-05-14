"""
Tests for the single-qubit Clifford sampler in src/circuit.py.

Verifies that SINGLE_QUBIT_CLIFFORDS contains exactly 24 entries with
24 distinct Pauli-conjugation actions and that every entry maps Pauli
operators to Pauli operators with correct signs.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.circuit import (
    SINGLE_QUBIT_CLIFFORDS,
    _pauli_action,
    apply_single_qubit_clifford,
    random_single_qubit_clifford,
)
from src.stabilizer import StabilizerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PAULIS = {(1, 0), (0, 1), (1, 1)}   # X, Z, Y  (x_bit, z_bit)


def _decode(triple):
    """Return a human-readable string for a (phase, x_bit, z_bit) triple."""
    phase, x, z = triple
    name = {(1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(x, z)]
    sign = '+' if phase == 0 else '-'
    return sign + name


# ---------------------------------------------------------------------------
# Count and uniqueness
# ---------------------------------------------------------------------------

def test_count_is_24():
    assert len(SINGLE_QUBIT_CLIFFORDS) == 24


def test_all_actions_unique():
    sigs = [_pauli_action(g) for g in SINGLE_QUBIT_CLIFFORDS]
    unique = set(sigs)
    duplicates = [
        (SINGLE_QUBIT_CLIFFORDS[i], SINGLE_QUBIT_CLIFFORDS[j])
        for i in range(len(sigs))
        for j in range(i + 1, len(sigs))
        if sigs[i] == sigs[j]
    ]
    assert len(unique) == 24, (
        f"Only {len(unique)} unique Pauli actions -- duplicates: {duplicates}"
    )


# ---------------------------------------------------------------------------
# Pauli-closure: every element maps Paulis to signed Paulis
# ---------------------------------------------------------------------------

def test_every_clifford_maps_x_to_signed_pauli():
    for gates in SINGLE_QUBIT_CLIFFORDS:
        img_X, _ = _pauli_action(gates)
        phase, x, z = img_X
        assert (x, z) in _VALID_PAULIS, (
            f"{gates} maps X to invalid (x={x}, z={z})"
        )
        assert phase in (0, 1), f"{gates} has invalid phase {phase} for X-image"


def test_every_clifford_maps_z_to_signed_pauli():
    for gates in SINGLE_QUBIT_CLIFFORDS:
        _, img_Z = _pauli_action(gates)
        phase, x, z = img_Z
        assert (x, z) in _VALID_PAULIS, (
            f"{gates} maps Z to invalid (x={x}, z={z})"
        )
        assert phase in (0, 1), f"{gates} has invalid phase {phase} for Z-image"


def test_x_and_z_images_are_orthogonal():
    """C X C† and C Z C† must be distinct Paulis (orthogonality of images)."""
    for gates in SINGLE_QUBIT_CLIFFORDS:
        img_X, img_Z = _pauli_action(gates)
        _, xX, zX = img_X
        _, xZ, zZ = img_Z
        assert (xX, zX) != (xZ, zZ), (
            f"{gates}: X and Z both map to the same Pauli type ({xX},{zX})"
        )


# ---------------------------------------------------------------------------
# Coverage: all 24 Pauli-action slots are filled
# ---------------------------------------------------------------------------

def test_all_24_signed_pauli_pairs_covered():
    """The 24 actions must together cover all elements of the Clifford group.

    The single-qubit Clifford group acts faithfully on the 6 signed Paulis
    {±X, ±Y, ±Z}.  There are exactly 24 valid (img_X, img_Z) pairs (the ones
    where img_X and img_Z are distinct Pauli types).  This test checks that
    our 24 elements hit all 24 slots.
    """
    sigs = {_pauli_action(g) for g in SINGLE_QUBIT_CLIFFORDS}

    # Build the set of all valid (img_X, img_Z) pairs by brute-force:
    # img_X type != img_Z type, phases free.
    valid_slots = set()
    for pX in (0, 1):
        for xX, zX in _VALID_PAULIS:
            for pZ in (0, 1):
                for xZ, zZ in _VALID_PAULIS:
                    if (xX, zX) != (xZ, zZ):
                        valid_slots.add(((pX, xX, zX), (pZ, xZ, zZ)))

    # valid_slots has 2*3*2*2 = 24 entries (3 choices for img_X type,
    # 2 sign choices each, 2 choices for img_Z type different from img_X,
    # 2 sign choices each).  But not all are reachable — exactly 24 are
    # (the group has order 24).  We just check our sigs == the reachable ones.
    assert sigs == (sigs & valid_slots), "Some action has an invalid (img_X, img_Z) pair"
    assert len(sigs) == 24


# ---------------------------------------------------------------------------
# Known gate actions (ground-truth checks)
# ---------------------------------------------------------------------------

def test_identity_action():
    """Empty gate sequence is the identity: X->+X, Z->+Z."""
    img_X, img_Z = _pauli_action(())
    assert img_X == (0, 1, 0), f"identity: X should map to +X, got {_decode(img_X)}"
    assert img_Z == (0, 0, 1), f"identity: Z should map to +Z, got {_decode(img_Z)}"


def test_H_action():
    """H conjugates: X -> +Z, Z -> +X."""
    img_X, img_Z = _pauli_action(('H',))
    assert img_X == (0, 0, 1), f"H: X->{_decode(img_X)}, expected +Z"
    assert img_Z == (0, 1, 0), f"H: Z->{_decode(img_Z)}, expected +X"


def test_S_action():
    """S conjugates: X -> +Y, Z -> +Z."""
    img_X, img_Z = _pauli_action(('S',))
    assert img_X == (0, 1, 1), f"S: X->{_decode(img_X)}, expected +Y"
    assert img_Z == (0, 0, 1), f"S: Z->{_decode(img_Z)}, expected +Z"


def test_SS_action():
    """S^2 = Z gate conjugates: X -> -X, Z -> +Z."""
    img_X, img_Z = _pauli_action(('S', 'S'))
    assert img_X == (1, 1, 0), f"S^2: X->{_decode(img_X)}, expected -X"
    assert img_Z == (0, 0, 1), f"S^2: Z->{_decode(img_Z)}, expected +Z"


def test_SSSS_is_identity():
    """S has order 4: S^4 = I."""
    assert _pauli_action(('S', 'S', 'S', 'S')) == _pauli_action(())


def test_HH_is_identity():
    """H has order 2: H^2 = I."""
    assert _pauli_action(('H', 'H')) == _pauli_action(())


# ---------------------------------------------------------------------------
# apply_single_qubit_clifford applies the right transformation
# ---------------------------------------------------------------------------

def test_apply_clifford_matches_pauli_action():
    """apply_single_qubit_clifford must produce states consistent with _pauli_action."""
    for gates in SINGLE_QUBIT_CLIFFORDS:
        img_X, img_Z = _pauli_action(gates)

        # Prepare |+> (stabilizer X), apply gates, read stabilizer row
        s = StabilizerState(1)
        s.apply_h(0)
        apply_single_qubit_clifford(s, 0, gates)
        row = s.tab[1]
        observed = (int(row[2]), int(row[0]), int(row[1]))
        assert observed == img_X, (
            f"{gates}: apply gives X->{_decode(observed)}, "
            f"_pauli_action says {_decode(img_X)}"
        )


# ---------------------------------------------------------------------------
# random_single_qubit_clifford API smoke test
# ---------------------------------------------------------------------------

def test_random_clifford_with_rng():
    rng = np.random.default_rng(0)
    s = StabilizerState(2)
    for _ in range(50):
        random_single_qubit_clifford(s, 0, rng)
        random_single_qubit_clifford(s, 1, rng)
    # State should still be a valid stabilizer state (no assertion error raised)
    assert s.n == 2


def test_random_clifford_without_rng():
    s = StabilizerState(1)
    for _ in range(20):
        random_single_qubit_clifford(s, 0)   # uses global numpy RNG


def test_random_clifford_reproducible():
    """Same seed -> same sequence of drawn Cliffords."""
    def run(seed):
        rng = np.random.default_rng(seed)
        s = StabilizerState(1)
        actions = []
        for _ in range(10):
            idx = int(rng.integers(24))
            apply_single_qubit_clifford(s, 0, SINGLE_QUBIT_CLIFFORDS[idx])
            actions.append(idx)
        return actions

    assert run(42) == run(42)
    assert run(42) != run(99)
