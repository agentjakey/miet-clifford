"""
Entanglement entropy for stabilizer states via GF(2) rank.

    S(A) = |A| - n + rank_GF2(M_B)

M_B is the n x 2|B| submatrix of stabilizer rows restricted to complement B's columns.
Formula verified by Bell state: n=2, A={0}, B={1}, rank(M_B)=2, S=1-2+2=1 ebit.

Reference: A. Hamma, R. Ionicioiu, P. Zanardi,
    "Bipartite entanglement and entropic boundary law in lattice spin systems,"
    Phys. Rev. A 71, 022315 (2005). doi:10.1103/PhysRevA.71.022315
"""
import numpy as np
from .stabilizer import StabilizerState


def _gf2_rank(matrix: np.ndarray) -> int:
    """Gaussian elimination over GF(2) to compute matrix rank.

    numpy.linalg.matrix_rank operates over the reals and cannot be used —
    it gives wrong results for binary matrices over GF(2).
    """
    M = matrix.copy().astype(np.int8)
    nrows, ncols = M.shape
    rank = 0
    pivot_row = 0
    for col in range(ncols):
        found = -1
        for row in range(pivot_row, nrows):
            if M[row, col] == 1:
                found = row
                break
        if found == -1:
            continue
        M[[pivot_row, found]] = M[[found, pivot_row]]
        for row in range(nrows):
            if row != pivot_row and M[row, col] == 1:
                M[row] ^= M[pivot_row]
        rank += 1
        pivot_row += 1
    return rank


def entanglement_entropy(state: StabilizerState, subsystem_A: list) -> float:
    """Bipartite entanglement entropy S(A) in ebits (log base 2).

    S(A) = |A| - n + rank_GF2(M_B)

    M_B restricts the n stabilizer rows to B's X and Z columns, where B is
    the complement of A. Using B columns (not A) is required: the rank counts
    generators with support entirely in B; the remaining generators cross the
    cut and determine the entropy. See Hamma, Ionicioiu, Zanardi (2005).
    """
    n = state.n
    A_set = set(subsystem_A)
    B = [j for j in range(n) if j not in A_set]
    if not B:
        return 0.0

    stab = state.tab[n:, :2*n]          # stabilizer rows, no phase column
    B_cols = [j for j in B] + [j + n for j in B]
    M_B = stab[:, B_cols]
    return float(len(subsystem_A) - n + _gf2_rank(M_B))


def half_chain_entropy(state: StabilizerState) -> float:
    """Entanglement entropy of left half [0, n//2) vs right half [n//2, n)."""
    return entanglement_entropy(state, list(range(state.n // 2)))
