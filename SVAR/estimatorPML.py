import numpy as np
import SVAR


def LogLike_t(u, b_vec, df_t, blocks=False,whiten=True):
    e = SVAR.SVARutil.innovation(u, b_vec, whiten=whiten, blocks=blocks)
    loglike = np.sum((-1) * ((1 - df_t) / 2) * np.log(1 + np.power(e, 2) / (df_t - 2)))
    return loglike

def PML_avar(n):
    # ToDo: V_est for PML
    V_est = np.full([n * n, n * n], np.nan)
    return V_est

# def LogLike_t_approx(u, b_vec, df_t, blocks=False,whiten=True):
#     e = SVAR.utilities.innovation(u, b_vec, whiten=whiten, blocks=blocks)
#     loglike = np.sum(((1 - df_t) / (2 * 2 * (df_t - 2))) * np.power(e, 4))
#     return loglike
#
# def LogLike_t_approx2(u, b_vec, df_t, blocks=False,whiten=True):
#     e = SVAR.utilities.innovation(u, b_vec, whiten=whiten, blocks=blocks)
#     loglike = np.sum(((1 - df_t) / (2 * 2 * (df_t - 2))) * np.power(e, 4) +
#                      ((1 - df_t) / (2 * 3 * (df_t - 2))) * np.power(e, 6) )
#     return loglike

def prepareOptions(u,
                   df_t = 7,
                   bstart=[], bstartopt='Rec',
                   blocks=False, n_rec=False,
                   printOutput=True,
                    estimator = 'PML'
                   ):
    options = dict()


    options['estimator'] = estimator

    T, n = np.shape(u)
    options['T'] = T
    options['n'] = n

    options['df_t'] = df_t

    options['printOutput'] = printOutput

    options['whiten'] = True
    _, V = SVAR.do_whitening(u, white=True)
    options['V'] = V

    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions=[])
    options['restrictions'] = restrictions
    options['blocks'] = blocks


    bstart = SVAR.estPrepare.prepare_bstart(estimator, bstart, u, options, bstartopt=bstartopt)
    options['bstart'] = bstart

    return options


def SVARout(est_SVAR, options, u):
    T, n = np.shape(u)
    out_SVAR = dict()
    out_SVAR['options'] = options

    b_est = est_SVAR['x']
    out_SVAR['b_est'] = b_est

    B_est = SVAR.get_BMatrix(b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                             blocks=options['blocks'])
    B_est = np.matmul(options['V'], B_est)
    out_SVAR['B_est'] = B_est

    e = SVAR.innovation(u, SVAR.get_BVector(B_est, restrictions=options['restrictions']),
                        restrictions=options['restrictions'],
                        blocks=options['blocks'])
    out_SVAR['e'] = e

    Omega_all = SVAR.SVARutil.get_Omega(e)
    out_SVAR['Omega_all'] = Omega_all
    omega = SVAR.SVARutil.get_Omega_Moments(e)
    out_SVAR['omega'] = omega

    out_SVAR['loss'] = est_SVAR['fun']

    V_est = PML_avar(options['n'])
    out_SVAR['Avar_est'] = V_est

    if options['printOutput']:
        SVAR.estOutput.print_out(n, T, out_SVAR)

    return out_SVAR












