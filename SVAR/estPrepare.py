import numpy as np
import SVAR
import SVAR.SVARutilGMM
import SVAR.estimatorPML
import SVAR.estimatorGMM
from SVAR.estimatorCholesky import get_B_Cholesky
from SVAR.SVARutil import get_BVector, get_block_rec, getRestrictions_blocks


# Prepare Boostrap
def prepare_bootstrapIRF(number_bootstrap=500, update_bstart=False, find_perm=True, path ='',
                         lags=1, add_const=True, add_trend=False, add_trend2=False,
                         irf_length=12,scale=False,parallel = False, num_cores_boot = 1,
                         seed_start = 0,   saveout = True  ):
    opt_bootstrap = dict()
    opt_bootstrap['number_bootstrap'] = number_bootstrap
    opt_bootstrap['update_bstart'] = update_bstart
    opt_bootstrap['find_perm'] = find_perm
    opt_bootstrap['path'] = path
    opt_bootstrap['parallel'] = parallel
    opt_bootstrap['num_cores_boot'] = num_cores_boot
    opt_bootstrap['seed_start'] = seed_start
    opt_bootstrap['saveout'] = saveout



    opt_redform = dict()
    opt_redform['lags'] = lags
    opt_redform['add_const'] = add_const
    opt_redform['add_trend'] = add_trend
    opt_redform['add_trend2'] = add_trend2

    opt_irf = dict()
    opt_irf['irf_length'] = irf_length
    opt_irf['scale'] = scale

    return opt_bootstrap, opt_redform, opt_irf


# Prepare opt_SVAR functions
def prepare_W(u, n, W, Wstartopt, moments, bstart, restrictions,   Wpara, S_func=False  ):
    if not (Wstartopt == 'I' or Wstartopt == 'WoptBstart' or Wstartopt == 'WoptparaNorm' or Wstartopt == 'specific'):
        raise ValueError('Unknown SVAR option Wstartopt.')
    if Wstartopt == 'I':
        W = np.eye(np.shape(moments)[0])
    elif Wstartopt == 'WoptBstart':
        W = SVAR.SVARutilGMM.get_W_opt(u, bstart, restrictions, moments,
                                       Wpara=Wpara, S_func=S_func)
    elif Wstartopt == 'WoptparaNorm':
        omega_normdist = np.array([0, 1, 0, 3, 0, 15])  # moments of normal distribution
        omega_normdist = np.tile(omega_normdist, (n, 1))
        W = np.linalg.inv(SVAR.SVARutilGMM.get_S_Indep(Moments_1=moments, Moments_2=moments, omega=omega_normdist ) )

    return W

def prepare_blocks_restrictions(n, n_rec, blocks, restrictions):
    # if no restrictions -> set options to no restrictions
    if not (n_rec) and blocks == False and np.array(restrictions).size == 0:
        restrictions = np.full([n, n], np.nan)
        # opt_SVAR['blocks'] == False

    # if n_rec specified -> overwrite blocks
    if not (n_rec == False):
        blocks = get_block_rec(n_rec, n)

    # if blocks specified -> overwrite restrictions
    if not (blocks == False):
        restrictions = getRestrictions_blocks(blocks)

    return restrictions, blocks

def prepare_moments(estimator, moments, addThirdMoments, addFourthMoments, moments_blocks, blocks, n, moments_MeanIndep=False):
    if np.array(moments).size == 0:
        if moments_blocks:
            blocks = blocks
        else:
            blocks = False
        moments = SVAR.SVARutilGMM.get_Moments(estimator, n, blocks=blocks,
                                               addThirdMoments=addThirdMoments,
                                               addFourthMoments=addFourthMoments, moments_MeanIndep=moments_MeanIndep)
    return moments

def prepare_bstart(estimator, bstart, u, options, bstartopt='Rec' ):
    if estimator == 'GMM' or estimator == 'CUE' or estimator == 'CSUE':
        if not (bstartopt == 'Rec' or bstartopt == 'CUEN' or bstartopt == 'I' or bstartopt == 'GMM_WF' or bstartopt == 'specific'):
            raise ValueError('Unknown SVAR option bstartopt.')
        if bstartopt == 'Rec':
            out = get_B_Cholesky(u)
            B_start = out['B_est']
            bstart = get_BVector(B_start, restrictions=options['restrictions'])
        if bstartopt == 'CUEN':
            T, n = np.shape(u)
            prepOptions_this = dict()
            prepOptions_this['printOutput'] = False
            prepOptions_this['blocks'] = options['blocks']
            prepOptions_this['restrictions'] = options['restrictions']
            prepOptions_this['moments'] = SVAR.SVARutilGMM.get_Moments('GMM', n, blocks=options['blocks'],
                                                         addThirdMoments=True,
                                                         addFourthMoments=True,
                                                         moments_MeanIndep=options['moments_MeanIndep'])
            prepOptions_this['Wpara'] = 'Independent'
            prepOptions_this['Avarparametric'] = 'Independent'
            prepOptions_this['Wstartopt'] = 'WoptBstart'
            prepOptions_this['bstartopt'] = 'GMM_WF'
            out = SVAR.SVARest(u, estimator='CUE', prepOptions=prepOptions_this)
            B_start = out['B_est']
            bstart = get_BVector(B_start, restrictions=options['restrictions'])
        elif bstartopt == 'I':
            n = np.shape(u)[1]
            B_start = np.eye(n)
            bstart = get_BVector(B_start, restrictions=options['restrictions'])
        elif bstartopt == 'GMM_WF':
            prepOptions_this = dict()
            prepOptions_this['printOutput'] = False
            prepOptions_this['blocks'] = options['blocks']
            out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions_this)
            B_start = out['B_est']
            bstart = get_BVector(B_start, restrictions=options['restrictions'])

    if estimator == 'GMM_W' or estimator == 'GMM_WF' or estimator == 'PML' or estimator == 'RidgeBWF':
        T, n = np.shape(u)
        if not (bstartopt == 'Rec' or bstartopt == 'specific' or bstartopt == 'GMM_WF'):
            raise ValueError('Unknown SVAR option bstartopt.')
        if bstartopt == 'Rec':
            if options['blocks']:
                b_counter = 0
                for block in options['blocks']:
                    n = block[1] - block[0] + 1
                    il1 = np.tril_indices(n, k=-1)
                    n_s = int(np.size(il1) / 2)
                    b_counter = b_counter + n_s
                bstart = np.zeros(b_counter)
            else:
                bstart = np.zeros(int(n * (n - 1) / 2))
        if bstartopt == 'GMM_WF':
            prepOptions_this = dict()
            prepOptions_this['printOutput'] = False
            prepOptions_this['blocks'] = options['blocks']
            out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions_this)
            bstart = out['b_est']


    return bstart