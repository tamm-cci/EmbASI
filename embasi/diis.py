from embasi.parallel_utils import root_print
import numpy as np

class DIIS():

    def __init__(self, init_update_matrix, hist_len=5, mixing_step=1.0, iter_mixing_start=3):

        self.hist_len = hist_len
        self.niter_tot = 0
        self.niter_restart = 0
        self.curr_hist_len = 0
        self.iter_mixing_start = iter_mixing_start
        self.mixing_step = mixing_step
        self.n_damped_mixing = 4

        self.update_matrix_hist = {}
        self.update_matrix_err_hist = {}

        self.add_history(init_update_matrix)

    def add_history(self, update_matrix):
        
        if len(self.update_matrix_hist) == self.hist_len:
            for idx in range(self.hist_len - 1):
                self.update_matrix_hist[idx] = self.update_matrix_hist[idx+1]
            self.update_matrix_hist.pop(self.hist_len - 1)

        if len(self.update_matrix_err_hist) == self.hist_len - 1:
            for idx in range(self.hist_len - 2):
                self.update_matrix_err_hist[idx] = self.update_matrix_err_hist[idx+1]
            self.update_matrix_err_hist.pop(self.hist_len - 2)

        curr_len = len(self.update_matrix_hist)
        self.update_matrix_hist[curr_len] = update_matrix.copy()

        if len(self.update_matrix_hist) > 1:
            for idx in range(len(self.update_matrix_hist) - 1):
                self.update_matrix_err_hist[idx] = self.update_matrix_hist[idx + 1] - self.update_matrix_hist[idx]

        if self.curr_hist_len < self.hist_len:
            self.curr_hist_len += 1

        root_print(f" CURR: {self.curr_hist_len}")
        root_print(f" HIST: {self.update_matrix_hist}")
        root_print(f" ERR HIST: {self.update_matrix_err_hist}")

    def diis_step(self, update_matrix):

        import scalapack4py.npscal.math_utils.operations as op

        self.niter_tot += 1

        if self.niter_tot < self.n_damped_mixing:
            curr_mixing_step = self.mixing_step
        else:
            curr_mixing_step = self.mixing_step

        self.add_history(update_matrix)

        # If only two matrices present, just add the 
        # residual between the two matrices scaled 
        # by the mixing step.
        if len(self.update_matrix_err_hist) == 1 or (self.iter_mixing_start > self.niter_tot):
            return ((1 - curr_mixing_step) * self.update_matrix_hist[self.curr_hist_len - 2]) + (curr_mixing_step * self.update_matrix_hist[self.curr_hist_len - 1])
        elif len(self.update_matrix_err_hist) < 1:
            raise Exception("DIIS ERROR: No update matrix history detected")

        solve_mat_size = len(self.update_matrix_err_hist) + 1
        solve_mat = np.zeros((solve_mat_size,solve_mat_size))
        
        for idx1 in range(solve_mat_size - 1):
            for idx2 in range(idx1 + 1):
                solve_mat[idx1, idx2] = op.trace(self.update_matrix_err_hist[idx1].T @ self.update_matrix_err_hist[idx2])

                if idx1 != idx2:
                    solve_mat[idx2, idx1] = solve_mat[idx1, idx2]

        # Assign 
        solve_mat[-1,:] = -1.0
        solve_mat[:,-1] = -1.0
        solve_mat[-1,-1] = 0
        root_print(f"SOLVE MAT PRE SOLVE: {solve_mat}")
        rhs = np.zeros(solve_mat_size)
        rhs[-1] = -1
        
        try:
            coeffs = np.linalg.solve(solve_mat, rhs)
        except:
            raise Exception("DIIS linalg solve failed.")

        root_print(f"SOLVE MAT AFTER SOLVE: {solve_mat}")
        root_print(f"COEFFS: {coeffs}")
        # Output extrpolated DIIS step
        #output = self.update_matrix_err_hist[idx] +
        for idx in range(solve_mat_size - 1):
            root_print(f"MIXING IDX: {idx}")
            if idx == 0:
                output = (coeffs[0] * self.update_matrix_hist[0])
            else:
                output = output + (coeffs[idx] * self.update_matrix_hist[idx])
                #output = output + (curr_mixing_step * coeffs[idx] * self.update_matrix_err_hist[idx])

        #output = ((1.0 - curr_mixing_step) * output) + (curr_mixing_step * self.update_matrix_hist[self.curr_hist_len - 1])

        return output

