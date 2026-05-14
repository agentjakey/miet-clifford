"""
Aaronson-Gottesman stabilizer tableau simulator for n-qubit Clifford circuits.

Reference: S. Aaronson and D. Gottesman,
    "Improved Simulation of Stabilizer Circuits,"
    Phys. Rev. A 70, 052328 (2004). arXiv:quant-ph/0406196

Tableau layout: (2n rows) x (2n+1 cols), dtype int8, all entries in {0,1}
  rows 0..n-1:    destabilizer generators
  rows n..2n-1:   stabilizer generators
  cols 0..n-1:    X part of each generator
  cols n..2n-1:   Z part of each generator
  col  2n:        phase bit (0 -> +1, 1 -> -1)

Gate operations (apply_h, apply_s, apply_cnot) are fully vectorized over rows
via numpy array slicing: O(n) per gate with no Python for-loop over rows.

Measurement implements both cases of A&G Algorithm 2 exactly, using mod-4
phase arithmetic in _rowsum to correctly track Pauli sign bookkeeping.
"""
import numpy as np


class StabilizerState:
    def __init__(self, n: int):
        self.n = n
        self.tab = np.zeros((2*n, 2*n+1), dtype=np.int8)
        for i in range(n):
            self.tab[i,   i]   = 1   # destabilizer X_i
            self.tab[n+i, n+i] = 1   # stabilizer   Z_i

    @property
    def tableau(self):
        """Alias for self.tab — exposes the canonical attribute name from A&G."""
        return self.tab

    def copy(self):
        s = StabilizerState(self.n)
        s.tab = self.tab.copy()
        return s

    def __repr__(self):
        lines = []
        for r in range(2*self.n):
            label = "D" if r < self.n else "S"
            row   = self.tab[r]
            pauli = "".join(["I", "X", "Z", "Y"][row[j] + 2*row[self.n+j]]
                            for j in range(self.n))
            phase = "+" if row[2*self.n] == 0 else "-"
            lines.append(f"{label}{r % self.n}: {phase}{pauli}")
        return "\n".join(lines)

    def apply_h(self, i):
        """Hadamard on qubit i. A&G Table 1: swap x,z; flip phase where both were 1."""
        n = self.n
        x_old = self.tab[:, i].copy()
        z_old = self.tab[:, n+i].copy()
        self.tab[:, 2*n] ^= x_old & z_old
        self.tab[:, i]    = z_old
        self.tab[:, n+i]  = x_old

    def apply_s(self, i):
        """Phase (S) gate on qubit i. A&G Table 1: phase ^= x & z_old; z ^= x."""
        n = self.n
        x     = self.tab[:, i]
        z_old = self.tab[:, n+i].copy()
        self.tab[:, 2*n] ^= x & z_old
        self.tab[:, n+i] ^= x

    def apply_cnot(self, ctrl, targ):
        """CNOT gate. A&G Table 1: phase, x_targ, z_ctrl updates."""
        n   = self.n
        x_c = self.tab[:, ctrl].copy()
        x_t = self.tab[:, targ].copy()
        z_c = self.tab[:, n+ctrl].copy()
        z_t = self.tab[:, n+targ].copy()
        self.tab[:, 2*n]    ^= x_c & z_t & (x_t ^ z_c ^ 1).astype(np.int8)
        self.tab[:, targ]   ^= x_c
        self.tab[:, n+ctrl] ^= z_t

    def _g(self, x1, z1, x2, z2):
        """Phase contribution g(x1,z1,x2,z2) from A&G Appendix. Returns int32 array."""
        result = np.zeros(len(x1), dtype=np.int32)
        mask11 = (x1 == 1) & (z1 == 1)
        result[mask11] = z2[mask11].astype(np.int32) - x2[mask11].astype(np.int32)
        mask10 = (x1 == 1) & (z1 == 0)
        result[mask10] = z2[mask10].astype(np.int32) * (2*x2[mask10].astype(np.int32) - 1)
        mask01 = (x1 == 0) & (z1 == 1)
        result[mask01] = x2[mask01].astype(np.int32) * (1 - 2*z2[mask01].astype(np.int32))
        return result

    def _rowsum(self, h, i):
        """Multiply row i into row h with mod-4 phase arithmetic. A&G Algorithm 1."""
        n   = self.n
        r_h = int(self.tab[h, 2*n])
        r_i = int(self.tab[i, 2*n])
        x_i = self.tab[i, :n]
        z_i = self.tab[i, n:2*n]
        x_h = self.tab[h, :n]
        z_h = self.tab[h, n:2*n]
        g   = self._g(x_i, z_i, x_h, z_h)
        tot = (2*r_h + 2*r_i + int(g.sum())) % 4
        self.tab[h, 2*n]    = np.int8(tot // 2)
        self.tab[h, :n]    ^= x_i
        self.tab[h, n:2*n] ^= z_i

    def measure(self, a, rng=None) -> int:
        """Measure qubit a in the Z basis. Returns 0 or 1. A&G Algorithm 2.

        Args:
            a:   qubit index
            rng: numpy.random.Generator for reproducible runs; uses global
                 numpy RNG when None (backward-compatible default).
        """
        n = self.n

        p = None
        for row in range(n, 2*n):
            if self.tab[row, a] == 1:
                p = row
                break

        if p is not None:
            # Case 1: random outcome
            for i in range(2*n):
                if i != p and self.tab[i, a] == 1:
                    self._rowsum(i, p)
            self.tab[p - n, :] = self.tab[p, :].copy()
            self.tab[p, :]     = 0
            self.tab[p, n + a] = 1
            outcome = int(rng.integers(2) if rng is not None else np.random.randint(0, 2))
            self.tab[p, 2*n]   = np.int8(outcome)
            return outcome

        # Case 2: deterministic outcome via scratch register.
        # Condition: destabilizer row i has X or Y on qubit a (x[i,a]=1).
        # The paired stabilizer n+i is the unique generator that doesn't commute
        # with D_i; accumulating those gives the product equal to ±Z_a.
        scratch = np.zeros(2*n + 1, dtype=np.int8)  # represents I, phase 0

        for i in range(n):
            if self.tab[i, a] == 1:  # destabilizer i anticommutes with Z_a
                r_s = int(scratch[2*n])
                r_i = int(self.tab[n+i, 2*n])
                x_i = self.tab[n+i, :n]
                z_i = self.tab[n+i, n:2*n]
                x_s = scratch[:n].copy()
                z_s = scratch[n:2*n].copy()
                g   = self._g(x_i, z_i, x_s, z_s)
                tot = (2*r_s + 2*r_i + int(g.sum())) % 4
                scratch[2*n]    = np.int8(tot // 2)
                scratch[:n]    ^= x_i
                scratch[n:2*n] ^= z_i

        return int(scratch[2*n])
