from embasi.parallel_utils import root_print
import numpy as np
import time

class DIIS():

    def __init__(self, init_update_matrix, hist_len=5, mixing_step=1.0, iter_mixing_start=4, debug=False):

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
                self.update_matrix_hist[idx] = self.update_matrix_hist[idx+1]
            self.update_matrix_hist.pop(self.hist_len - 1)

        if len(self.update_matrix_err_hist) == self.hist_len:
            for idx in range(self.hist_len - 1):
                self.update_matrix_err_hist[idx] = self.update_matrix_err_hist[idx+1]
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
            self.solve_mat[idx1, solve_mat_size-2] = op.trace(self.update_matrix_err_hist[idx1].T @ self.update_matrix_err_hist[solve_mat_size-2])
            self.solve_mat[solve_mat_size-2, idx1] = self.solve_mat[idx1, solve_mat_size-2]
        root_print(f"Tome for Coeff Calc Matmul: {time.time()-time_s}")
        # Assign
        self.solve_mat[-1,:] = -1.0
        self.solve_mat[:,-1] = -1.0
        self.solve_mat[-1,-1] = 0

        if self.debug: root_print(f"SOLVE MAT: {self.solve_mat}")

        rhs = np.zeros(solve_mat_size)
        rhs[-1] = -1

        #w, v = scipy.linalg.eigh(solve_mat)

        #if np.any(abs(w)<1e-14):
        #    root_print(f"Linear dependence in DIIS error vectors")
        #    idx = abs(w)>1e-14
        #    coeffs = np.dot(v[:,idx]*(1./w[idx]), np.dot(v[:,idx].T.conj(), rhs))
        #else:
        time_s = time.time()
        try:
            import scipy
            #coeffs = np.linalg.solve(solve_mat, rhs)
            coeffs = scipy.optimize.lsq_linear(self.solve_mat, rhs)
            coeffs = coeffs.x
        except:
            raise Exception("DIIS linalg solve failed.")
        root_print(f"Tome for Coeff Leat squares (if this is the slow step, I am going to flip out): {time.time()-time_s}")
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
            return output
        else:
            time_s = time.time()
            self.add_history(update_matrix, residual)
            root_print(f"Time for history add: {time.time()-time_s}")

        time_s = time.time()            
        if coeff_mat is None:
            coeffs = self.get_coeff_matrix()
        else:
            coeffs = coeff_mat
        root_print(f"Tome for Coeff Calc: {time.time()-time_s}")

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
        root_print(f"Time for extrapolation: {time.time()-time_s}")

        self.prev_opt_in = output.copy()

        return output

