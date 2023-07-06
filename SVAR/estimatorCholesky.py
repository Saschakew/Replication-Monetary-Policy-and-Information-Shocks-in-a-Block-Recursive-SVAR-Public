import numpy as np
import SVAR


def get_B_Cholesky(u):
    T, n = np.shape(u)
    Sigma = np.dot(np.transpose(u), u) / T
    B = np.linalg.cholesky(Sigma)
    e = np.transpose(np.matmul(np.linalg.inv(B), np.transpose(u)))
    Omega = SVAR.SVARutil.get_Omega(e)

    out = dict()
    out['B_est'] = B
    out['b_est'] = SVAR.get_BVector(B)
    out['e'] = e
    out['loss'] = 0
    out['Omega_all'] = Omega
    out['options'] = dict()
    return out