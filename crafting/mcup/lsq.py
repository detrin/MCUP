"""Basic interface for least-square minimization."""

from scipy.optimize import least_squares
import numpy as np

class LeastSquares(object):
    '''Core object for LSQ evaluation with different solvers.'''

    def __init__(self, lsq_type):
        '''Switching method for different LSQ solvers.'''
        if lsq_type not in ['scipy']:
            raise ValueError("`lsq_type` must be 'scipy'.")

        if lsq_type == "scipy":
            # Docs are here https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            self.lsq_solver = least_squares
    
    def set_params(
            self, jac='2-point', bounds=(-np.inf, np.inf), method='trf',
            ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear',
            f_scale=1.0, diff_step=None, tr_solver=None, tr_options={},
            jac_sparsity=None, max_nfev=None, verbose=0):
        '''Setting and checking input parameters for different solvers.'''
        self.jac = jac
        self.bounds = bounds
        self.method = method
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.x_scale = x_scale
        self.loss = loss
        self.f_scale = f_scale
        self.diff_step = diff_step
        self.tr_solver = tr_solver
        self.tr_options = tr_options
        self.jac_sparsity = jac_sparsity
        self.max_nfev = max_nfev
        self.verbose = verbose

        return True
    
    def fit(self, fun, x, y, w_0):
        '''Fit function using LSQ solver.'''
        def fun_diff(w):
            return fun(x, *w) - y

        lsq_result = self.lsq_solver(
                fun_diff, w_0,
                jac=self.jac, bounds=self.bounds, method=self.method, ftol=self.ftol, 
                xtol=self.xtol, gtol=self.gtol, x_scale=self.x_scale, loss=self.loss, 
                f_scale=self.f_scale, diff_step=self.diff_step, tr_solver=self.tr_solver, 
                tr_options=self.tr_options, jac_sparsity=self.jac_sparsity, 
                max_nfev=self.max_nfev, verbose=self.verbose
            )
        J = lsq_result.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))
        return [lsq_result.x, var]