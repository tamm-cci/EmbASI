from embasi.parallel_utils import root_print
import numpy as np
import time

class DIIS():

    def __init__(self, init_update_matrix, hist_len=5, mixing_step=1.0, iter_mixing_start=8, debug=False):

        self.hist_len = hist_len
        self.niter_tot = 0
        self.niter_restart = 0
        self.curr_hist_len = 0
        self.iter_mixing_start = iter_mixing_start
        self.mixing_step = mixing_step
        self.n_damped_mixing = 4

        self.update_matrix_hist = {}
        self.update_matrix_err_hist = {}

        self.prev_opt_in = init_update_matrix

        self.debug = debug

    def add_history(self, update_matrix, update_matrix_residual):
        
        if len(self.update_matrix_hist) == self.hist_len:
            for idx in range(self.hist_len - 1):
                self.update_matrix_hist[idx] = self.update_matrix_hist[idx+1].copy()
            self.update_matrix_hist.pop(self.hist_len - 1)

        if len(self.update_matrix_err_hist) == self.hist_len:
            for idx in range(self.hist_len - 1):
                self.update_matrix_err_hist[idx] = self.update_matrix_err_hist[idx+1].copy()
            self.update_matrix_err_hist.pop(self.hist_len - 1)

        curr_len = len(self.update_matrix_hist)
        self.update_matrix_hist[curr_len] = update_matrix.copy()
        self.update_matrix_err_hist[curr_len] = update_matrix_residual.copy()

        if self.curr_hist_len < self.hist_len:
            self.curr_hist_len += 1

    def get_coeff_matrix(self):

        import scalapack4py.npscal.math_utils.operations as op

        if len(self.update_matrix_err_hist) == 0:
            return None
        
        solve_mat_size = len(self.update_matrix_err_hist) + 1

        if not hasattr(self, "solve_mat"):
            self.solve_mat = np.zeros((solve_mat_size,solve_mat_size))

        if len(self.update_matrix_err_hist) == self.hist_len:
            if np.shape(self.solve_mat)[0] != solve_mat_size:
                temp_solv_mat = np.zeros((solve_mat_size,solve_mat_size))
                old_smat_size = np.shape(self.solve_mat)[0]
                temp_solv_mat[0:old_smat_size, 0:old_smat_size] = self.solve_mat
                self.solve_mat = temp_solv_mat
            else:
                self.solve_mat = np.roll(self.solve_mat, shift=-1, axis=(0,1))
        else:
            temp_solv_mat = np.zeros((solve_mat_size,solve_mat_size))
            temp_solv_mat[0:solve_mat_size-2, 0:solve_mat_size-2] = self.solve_mat[0:solve_mat_size-2, 0:solve_mat_size-2]
            self.solve_mat = temp_solv_mat

        time_s = time.time()
        for idx1 in range(solve_mat_size - 1):
            # Tr(E_i^T E_j) = sum_ij E_i * E_j: O(n^2), no matrix product.
            self.solve_mat[idx1, solve_mat_size-2] = np.einsum(
                "ij,ij->", self.update_matrix_err_hist[idx1], self.update_matrix_err_hist[solve_mat_size-2])
            self.solve_mat[solve_mat_size-2, idx1] = self.solve_mat[idx1, solve_mat_size-2]
        if self.debug: root_print(f"Time for Coeff Calc Matmul: {time.time()-time_s}")
        # Assign
        self.solve_mat[-1,:] = -1.0
        self.solve_mat[:,-1] = -1.0
        self.solve_mat[-1,-1] = 0

        if self.debug: root_print(f"SOLVE MAT: {self.solve_mat}")

        rhs = np.zeros(solve_mat_size)
        rhs[-1] = -1

        time_s = time.time()
        try:
            import scipy
            coeffs = scipy.optimize.lsq_linear(self.solve_mat, rhs)
            coeffs = coeffs.x
        except:
            raise Exception("DIIS linalg solve failed.")
        return coeffs

    def diis_step(self, update_matrix, coeff_mat=None):

        import scalapack4py.npscal.math_utils.operations as op
        import time

        self.niter_tot += 1

        if self.niter_tot <= self.n_damped_mixing:
            curr_mixing_step = self.mixing_step
        else:
            curr_mixing_step = self.mixing_step

        residual = update_matrix - self.prev_opt_in

        # If only two matrices present, just add the 
        # residual between the two matrices scaled 
        # by the mixing step.
        if (self.iter_mixing_start > self.niter_tot):
            output = self.prev_opt_in + (curr_mixing_step * residual)
            self.prev_opt_in = output.copy()
            return output, None
        else:
            time_s = time.time()
            # Pulay/Anderson mixing extrapolates the *input* matrices,
            # sum_i c_i (in_i + alpha * R_i) with R_i = out_i - in_i.
            # Storing the output here instead gave sum_i c_i (in_i + (1 + alpha) R_i).
            self.add_history(self.prev_opt_in, residual)
            if self.debug: root_print(f"Time for history add: {time.time()-time_s}")

        time_s = time.time()            
        if coeff_mat is None:
            coeffs = self.get_coeff_matrix()
        else:
            coeffs = coeff_mat

        if self.debug: root_print(f"Time for Coeff Calc: {time.time()-time_s}")
        if self.debug: root_print(f"COEFFS: {coeffs}")

        # Output extrpolated DIIS step
        time_s = time.time()
        for idx in range(len(coeffs)-1):
            if idx == 0:
                #output = (coeffs[0] * self.update_matrix_hist[0])
                output = coeffs[0] * (self.update_matrix_hist[0] + (curr_mixing_step * self.update_matrix_err_hist[0]))
            else:
                #output = output + (coeffs[idx] * self.update_matrix_hist[idx])
                curr_it = self.update_matrix_hist[idx].copy()
                curr_it *= coeffs[idx]
                curr_his = self.update_matrix_err_hist[idx].copy()
                curr_his *= curr_mixing_step * coeffs[idx]
                output += curr_it
                output += curr_his
                #output += (coeffs[idx] * (self.update_matrix_hist[idx] + (curr_mixing_step * self.update_matrix_err_hist[idx])))
        if self.debug: root_print(f"Time for extrapolation: {time.time()-time_s}")

        self.prev_opt_in = output.copy()

        return output, coeffs



class CommutatorDIIS():
    """Pulay commutator DIIS over several coupled Fock matrices

    Extrapolates a set of Fock matrices (e.g. the embedded Fock matrices
    of subsystems A and B in freeze-and-thaw) with one shared set of
    coefficients. The error vector of each Fock matrix is its commutator
    e = F D S - S D F, and the DIIS error is the concatenation of all
    of them, which is zero only when every subsystem is converged
    together. Each entry of a SpinKpointArray (spin, k-point) is a
    separate block of the error vector.

    Blocks may be dense np.ndarray (serial) or distributed NPScal
    (parallel) arrays. With NPScal, every inner product is reduced over
    the whole BLACS grid, so all ranks build the same B matrix and
    obtain the same coefficients.

    The error inner products <e_i|e_j> are stored between updates, so
    each update computes only the n_hist products of the newest error
    vector, not the full n_hist (n_hist + 1) / 2.
    """

    def __init__(self, hist_len=8):
        self.hist_len = hist_len
        self.fock_hist = []
        self.err_hist = []
        # Unscaled <e_i|e_j> for the errors in err_hist, in the same order
        self.err_prod = np.zeros((0, 0))

    @staticmethod
    def _inner(errs_1, errs_2):
        """Re <errs_1|errs_2>, summed over subsystems and (spin, k-point)
        blocks, each k-point weighted. Written as sum(conj(a) * b) rather than np.vdot, which
        NPScal does not implement; np.sum of an NPScal is reduced over the
        whole grid, so the result is the same global value on every rank."""
        from embasi.kpoint_parallel import gather_by_key
        local = {}
        for i_sub, (err_1, err_2) in enumerate(zip(errs_1, errs_2)):
            for (ispin, ikpt) in err_1.keys():
                # k-points count by their Brillouin-zone weight (1 at Gamma)
                local[(i_sub, ispin, ikpt)] = err_1.kpt_weights[ikpt] * \
                    float(np.real(np.sum(np.conj(err_1.data[(ispin, ikpt)]) *
                                         err_2.data[(ispin, ikpt)])))
        if errs_1[0].n_kpoints > 1:
            # Blocks of other process groups too (collective; decided on the
            # global k-point count)
            local = gather_by_key(local)
        total = 0.0
        for key in sorted(local):
            total += local[key]
        return total

    def update(self, focks, errs):
        """Adds focks/errs (lists of SpinKpointArray) to the history and
        returns the extrapolated Fock matrices, in the same order"""
        self.fock_hist.append([fock.copy() for fock in focks])
        self.err_hist.append([err.copy() for err in errs])

        # Grow the stored products by the newest row/column only
        n_old = self.err_prod.shape[0]
        new_row = np.array([self._inner(self.err_hist[i], self.err_hist[-1])
                            for i in range(n_old + 1)])
        err_prod = np.zeros((n_old + 1, n_old + 1))
        err_prod[:n_old, :n_old] = self.err_prod
        err_prod[n_old, :] = err_prod[:, n_old] = new_row
        self.err_prod = err_prod

        if len(self.fock_hist) > self.hist_len:
            self.fock_hist.pop(0)
            self.err_hist.pop(0)
            self.err_prod = self.err_prod[1:, 1:]

        n_hist = len(self.fock_hist)
        if n_hist == 1:
            return [fock.copy() for fock in focks]

        b_mat = np.zeros((n_hist + 1, n_hist + 1))
        # Scale for conditioning: the coefficients are scale invariant
        b_mat[:n_hist, :n_hist] = self.err_prod / np.max(np.abs(np.diag(self.err_prod)))
        b_mat[n_hist, :n_hist] = b_mat[:n_hist, n_hist] = -1.0
        rhs = np.zeros(n_hist + 1)
        rhs[n_hist] = -1.0

        try:
            coeffs = np.linalg.solve(b_mat, rhs)[:n_hist]
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(b_mat, rhs, rcond=None)[0][:n_hist]

        extrapolated = []
        for i_fock in range(len(focks)):
            fock = self.fock_hist[0][i_fock] * float(coeffs[0])
            for i_hist in range(1, n_hist):
                fock = fock + self.fock_hist[i_hist][i_fock] * float(coeffs[i_hist])
            extrapolated.append(fock)

        return extrapolated
