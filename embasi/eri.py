"""AO-basis electron repulsion integrals exported by a QM code

The QM code exports 4-index AO ERIs once (e.g. SIESTA's QComp ERIs through
the ASI ERI callback). :class:`AOERIs` keeps them in PySCF's 4-fold packed
layout and provides what a correlated solver built on the embedded
Hamiltonian needs: Coulomb/exchange matrices for a density matrix and the
integrals transformed to a set of orbitals.

Conventions (as exported, see the ASI ERI descriptor): chemists' notation
(ij|kl), AO basis of the QM code (its own orbital order), Hartree. In the
4-fold packed layout row/column p = i*(i+1)/2 + j (i >= j) indexes the pair
(ij), the same packing as ``pyscf.ao2mo.restore(4, ...)``.

Density matrices are spin-summed (2-occupancy) unless a method says
otherwise, as everywhere else in EmbASI.
"""
import numpy as np
# Imported at module load, before any ASI library: SIESTA's library brings its
# own netCDF/HDF5, and once it is loaded RTLD_GLOBAL its HDF5 symbols shadow
# those h5py (imported by PySCF) was built against, so a later first import of
# PySCF fails in h5py ("Not a datatype").
from pyscf import ao2mo  # noqa: F401
from pyscf.scf import hf  # noqa: F401

# ASI ERI descriptor values (asi.h); taken from asi4py when it has them
try:
    from asi4py.pyasi import ASI_ERI_NOTATION_CHEMIST, ASI_ERI_IDX_4, \
        ASI_ERI_BASIS_AO, ASI_ERI_SYM_1, ASI_ERI_SYM_4, ASI_ERI_SYM_8
except ImportError:  # asi4py without ERI support
    ASI_ERI_NOTATION_CHEMIST = 1
    ASI_ERI_IDX_4 = 0
    ASI_ERI_BASIS_AO = 0
    ASI_ERI_SYM_1 = 0
    ASI_ERI_SYM_4 = 1
    ASI_ERI_SYM_8 = 2


def _npair(n):
    return n * (n + 1) // 2


class AOERIs:
    """4-index AO ERIs in PySCF's 4-fold packed layout

    Parameters
    ----------
    eri_s4: np.ndarray
        (npair, npair) array, npair = nao*(nao+1)/2, chemists' notation,
        Hartree
    nao: int
        Number of AO basis functions
    """

    def __init__(self, eri_s4, nao):
        eri_s4 = np.ascontiguousarray(eri_s4, dtype=float)
        npair = _npair(nao)
        if eri_s4.shape != (npair, npair):
            raise ValueError(
                f"4-fold packed ERIs for nao={nao} must have shape "
                f"{(npair, npair)}; got {eri_s4.shape}")
        self.eri_s4 = eri_s4
        self.nao = int(nao)

    @classmethod
    def from_asi(cls, eri, descr, nao):
        """Builds from an ERI matrix and descriptor as passed to the ASI ERI callback

        Parameters
        ----------
        eri: np.ndarray
            2-D ERI matrix of asi4py's ``eri_shape()``
        descr: dict
            ASI_eri_matrix_descr fields (``sum_notation``, ``n_index``,
            ``basis_type``, ``eri_symmetry``)
        nao: int
            Number of AO basis functions
        """
        if descr["n_index"] != ASI_ERI_IDX_4:
            raise NotImplementedError("Only 4-index ERIs are supported")
        if descr["basis_type"] != ASI_ERI_BASIS_AO:
            raise NotImplementedError("Only AO-basis ERIs are supported")
        eri = np.asarray(eri, dtype=float)
        sym = descr["eri_symmetry"]

        if sym == ASI_ERI_SYM_1:
            # Pair index i + n*j: the Fortran-order flattening of (i, j)
            full = eri.reshape((nao, nao, nao, nao), order='F')
            if descr["sum_notation"] != ASI_ERI_NOTATION_CHEMIST:
                # <ij|kl> = (ik|jl)
                full = full.transpose(0, 2, 1, 3)
            from pyscf import ao2mo
            return cls(ao2mo.restore(4, np.ascontiguousarray(full), nao), nao)

        if descr["sum_notation"] != ASI_ERI_NOTATION_CHEMIST:
            raise ValueError("Pair-packed ERIs must be in chemists' notation")
        if sym == ASI_ERI_SYM_4:
            return cls(eri, nao)
        if sym == ASI_ERI_SYM_8:
            # One triangle of the symmetric (npair, npair) matrix is stored
            lower = np.tril(eri)
            return cls(lower + np.tril(eri, -1).T, nao)
        raise ValueError(f"Unsupported ERI symmetry {sym}")

    # ------------------------------------------------------------------
    # Integrals
    # ------------------------------------------------------------------

    def tensor(self):
        """Full (nao, nao, nao, nao) tensor (ij|kl)"""
        from pyscf import ao2mo
        return ao2mo.restore(1, self.eri_s4, self.nao)

    def eri_mo(self, mo_coeff):
        """(pq|rs) over the orbitals ``mo_coeff`` (nao, nmo), shape (nmo,)*4"""
        from pyscf import ao2mo
        mo_coeff = np.asarray(mo_coeff, dtype=float)
        nmo = mo_coeff.shape[1]
        return ao2mo.restore(1, ao2mo.incore.full(self.eri_s4, mo_coeff), nmo)

    def eri_mo_mixed(self, mo_a, mo_b):
        """(p_a q_a|r_b s_b) for two orbital sets of equal size, shape (nmo,)*4"""
        from pyscf import ao2mo
        mo_a = np.asarray(mo_a, dtype=float)
        mo_b = np.asarray(mo_b, dtype=float)
        nmo = mo_a.shape[1]
        if mo_b.shape[1] != nmo:
            raise ValueError(
                f"Orbital sets must have the same size; got {nmo} and {mo_b.shape[1]}")
        out = ao2mo.incore.general(self.eri_s4, (mo_a, mo_a, mo_b, mo_b),
                                   compact=False)
        return np.asarray(out).reshape(nmo, nmo, nmo, nmo)

    # ------------------------------------------------------------------
    # Mean-field potentials
    # ------------------------------------------------------------------

    def get_jk(self, dm):
        """(J[dm], K[dm]) for a symmetric AO density matrix

        J_ij = sum_kl (ij|kl) dm_kl, K_ij = sum_kl (ik|jl) dm_kl. Linear in
        dm, so the spin-summed or a single spin channel's density can be
        passed.
        """
        from pyscf.scf import hf
        dm = np.asarray(dm, dtype=float)
        if dm.shape != (self.nao, self.nao):
            raise ValueError(f"Density matrix must be ({self.nao}, {self.nao}); got {dm.shape}")
        vj, vk = hf.dot_eri_dm(self.eri_s4, dm, hermi=1)
        return np.asarray(vj), np.asarray(vk)

    def get_j(self, dm):
        return self.get_jk(dm)[0]

    def get_k(self, dm):
        return self.get_jk(dm)[1]

    def veff_hf(self, dm):
        """Restricted Hartree-Fock potential J - K/2 of a 2-occupancy dm"""
        vj, vk = self.get_jk(dm)
        return vj - 0.5 * vk

    def veff_uhf(self, dm_a, dm_b):
        """Unrestricted HF mean field of a 1-occupancy pair

        Returns (v_alpha, v_beta, e_two_body) with v_s = J[dm_a+dm_b] - K[dm_s]
        and e_two_body = 0.5 tr[dm_a v_alpha] + 0.5 tr[dm_b v_beta].
        """
        dm_a = np.asarray(dm_a, dtype=float)
        dm_b = np.asarray(dm_b, dtype=float)
        j_tot = self.get_j(dm_a + dm_b)
        v_a = j_tot - self.get_k(dm_a)
        v_b = j_tot - self.get_k(dm_b)
        e_two_body = 0.5 * (np.einsum("ij,ji->", dm_a, v_a)
                            + np.einsum("ij,ji->", dm_b, v_b))
        return v_a, v_b, float(e_two_body)

    def hartree_energy(self, dm):
        """1/2 tr[dm J[dm]] of a 2-occupancy dm"""
        dm = np.asarray(dm, dtype=float)
        return 0.5 * float(np.einsum("ij,ji->", dm, self.get_j(dm)))

    def exchange_energy(self, dm):
        """Restricted HF exchange energy -1/4 tr[dm K[dm]] of a 2-occupancy dm"""
        dm = np.asarray(dm, dtype=float)
        return -0.25 * float(np.einsum("ij,ji->", dm, self.get_k(dm)))
