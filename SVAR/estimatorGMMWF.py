import numpy as np
import SVAR
from SVAR.SVARutilGMM import get_Moments,  get_S_uncentered, get_g_wf,get_G_Indep, get_S_Indep,get_S
import SVAR.estimatorGMMW
import scipy.stats

def loss(u, b, restrictions, moments, moments_powerindex, blocks=False):
    whiten = True
    g = get_g_wf(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks)
    Q = - np.sum(np.power(g, 2))
    return Q

def loss_MIcorrection(u,b,restrictions,moments, moments_powerindex, blocks=False):
    whiten = True
    g = get_g_wf(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks )
    Q = + 6* np.sum(np.power(g, 2))

    return Q

def gradient(u, b, Jacobian, W, restrictions, moments, moments_powerindex):
    # e = SVAR.innovation(u, b, restrictions=restrictions)
    # CoOmega = SVAR.SVARutil.get_CoOmega(e)
    #
    # Jac_temp = Jacobian(u=u, b=b, restrictions=restrictions, CoOmega=CoOmega)
    # this_g = get_g(b=b[:], u=u[:], restrictions=restrictions[:],
    #                moments=moments[:], moments_powerindex=moments_powerindex)
    # dGMM_temp  =  2 * np.linalg.multi_dot([np.transpose(Jac_temp), W, this_g])
    return []

def get_AVAR(n, G2,GC, S22, S2C, SC2, SCC, Wfast, restrictions=[] , blocks=[] ):
    # ToDo: only works with default moments
    V = SVAR.estimatorGMMW.get_Avar(n, G2,GC, S22, S2C, SC2, SCC, W=Wfast,restrictions=restrictions)

    return V


def prepareOptions(u,
                   addThirdMoments=True, addFourthMoments=True, moments=[], moments_blocks=True, moments_MeanIndep=False,
                   bstart=[], bstartopt='Rec',
                   Avarparametric='Uncorrelated',
                   blocks=False, n_rec=False,
                   printOutput=True,
                   estimator = 'GMM_WF',
                   Bcetenter=[]
                   ):
    options = dict()


    options['estimator'] = estimator

    T, n = np.shape(u)
    options['T'] = T
    options['n'] = n

    options['printOutput'] = printOutput

    options['whiten'] = True
    _, V = SVAR.do_whitening(u, white=True)
    options['V'] = V


    if Avarparametric == 'Uncorrelated' or Avarparametric == 'Uncorrelated_uncentered' or Avarparametric == 'Independent':
        options['Avarparametric'] = Avarparametric
    else:
        print("Invalid Avarparametric value. Set Avarparametric=Uncorrelated")
        options['Avarparametric'] = 'Uncorrelated'

    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions=[])
    options['restrictions'] = restrictions
    options['blocks'] = blocks

    options['addThirdMoments'] = addThirdMoments
    options['addFourthMoments'] = addFourthMoments
    moments = SVAR.estPrepare.prepare_moments(estimator, moments, addThirdMoments, addFourthMoments, moments_blocks,
                                              blocks, n)
    options['moments'] = moments
    options['moments_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(moments)

    options['moments_MeanIndep'] = moments_MeanIndep
    if moments_MeanIndep:
        options['moments_MIcorrection'] = SVAR.SVARutilGMM.get_Moments_MIcorrection(n, blocks=blocks)
        options['moments_MIcorrection_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(options['moments_MIcorrection'])

    bstart = SVAR.estPrepare.prepare_bstart(estimator, bstart, u, options, bstartopt=bstartopt)
    options['bstart'] = bstart

    options['Bcetenter'] = Bcetenter


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

    e = SVAR.innovation(u, SVAR.get_BVector(B_est, restrictions=options['restrictions'],blocks=options['blocks']), restrictions=options['restrictions'] ,
                        blocks=options['blocks'])
    out_SVAR['e'] = e

    Omega_all = SVAR.SVARutil.get_Omega(e)
    out_SVAR['Omega_all'] = Omega_all
    omega = SVAR.SVARutil.get_Omega_Moments(e)
    CoOmega = SVAR.SVARutil.get_CoOmega(e)
    out_SVAR['omega'] = omega

    z, options['V'] = SVAR.do_whitening(u, white=True)
    out_SVAR['loss'] = loss(z, b_est, options['restrictions'], options['moments'], options['moments_powerindex'], blocks=options['blocks'])
    if options['moments_MeanIndep']:
        out_SVAR['loss'] = out_SVAR['loss'] - SVAR.estimatorGMMWF.loss_MIcorrection(z,b_est,restrictions=options['restrictions'],
                                                                  moments=options['moments_MIcorrection'],
                                                                  moments_powerindex=options['moments_MIcorrection_powerindex'], blocks=options['blocks'])

    Moments2 = SVAR.SVARutilGMM.get_Mr(2, n)
    Moments2 = np.append(Moments2, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    comoments = get_Moments('GMM_W', n, blocks= options['blocks'],
                            addThirdMoments= options['addThirdMoments'],
                            addFourthMoments= options['addFourthMoments'],moments_MeanIndep=options['moments_MeanIndep'])
    Wfast = SVAR.estimatorGMMW.get_W_fast(comoments)
    if options['Avarparametric'] == 'Independent':
        G2 = get_G_Indep(Moments2, out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])
        GC = get_G_Indep(comoments, out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])

        S22 = get_S_Indep(Moments2, Moments2, out_SVAR['omega'])
        S2C = get_S_Indep(Moments2, comoments, out_SVAR['omega'])
        SC2 = get_S_Indep(comoments, Moments2, out_SVAR['omega'])
        SCC = get_S_Indep(comoments, comoments, out_SVAR['omega'])
    elif options['Avarparametric'] == 'Uncorrelated_uncentered':
        Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=comoments, restrictions=options['restrictions'])
        Jacobian2 = SVAR.SVARutilGMM.generate_Jacobian_function(moments=Moments2, restrictions=options['restrictions'])

        GC = Jacobian(u=u, b= SVAR.get_BVector(B_est,restrictions=options['restrictions']) , restrictions=options['restrictions'], CoOmega=CoOmega)
        G2 = Jacobian2(u=u, b= SVAR.get_BVector(B_est,restrictions=options['restrictions']) , restrictions=options['restrictions'], CoOmega=CoOmega)

        if np.shape(comoments)[0] == 0:
            GC = get_G_Indep(comoments, out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])

        S = get_S_uncentered(u, SVAR.get_BVector(B_est, restrictions=np.full([n, n], np.nan)),
                             np.append(Moments2, options['moments'], axis=0), np.full([n, n], np.nan))
        S22 = S[:np.shape(Moments2)[0],:np.shape(Moments2)[0]]
        S2C = S[:np.shape(Moments2)[0],np.shape(Moments2)[0]:]
        SC2 = S[np.shape(Moments2)[0]:,:np.shape(Moments2)[0]]
        SCC = S[np.shape(Moments2)[0]:,np.shape(Moments2)[0]:]
    elif options['Avarparametric'] == 'Uncorrelated':
        Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=comoments, restrictions=options['restrictions'])
        Jacobian2 = SVAR.SVARutilGMM.generate_Jacobian_function(moments=Moments2, restrictions=options['restrictions'])

        GC = Jacobian(u=u, b= SVAR.get_BVector(B_est,restrictions=options['restrictions']) , restrictions=options['restrictions'], CoOmega=CoOmega)
        G2 = Jacobian2(u=u, b= SVAR.get_BVector(B_est,restrictions=options['restrictions']) , restrictions=options['restrictions'], CoOmega=CoOmega)

        if np.shape(comoments)[0] == 0:
            GC = get_G_Indep(comoments, out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])

        S = get_S(u, SVAR.get_BVector(B_est,restrictions=np.full([n, n], np.nan)),  np.append(Moments2, comoments, axis=0) , np.full([n, n], np.nan) )
        S22 = S[:np.shape(Moments2)[0],:np.shape(Moments2)[0]]
        S2C = S[:np.shape(Moments2)[0],np.shape(Moments2)[0]:]
        SC2 = S[np.shape(Moments2)[0]:,:np.shape(Moments2)[0]]
        SCC = S[np.shape(Moments2)[0]:,np.shape(Moments2)[0]:]

    V_est = get_AVAR(n, G2,GC, S22, S2C, SC2, SCC ,restrictions=options['restrictions'] , Wfast=Wfast)
    out_SVAR['Avar_est'] = V_est


    b = SVAR.get_BVector(B_est, restrictions=options['restrictions'], whiten=False)
    t_all = np.empty_like(b)
    t_all_p = np.empty_like(b)
    for idx, bthis in np.ndenumerate(b):
        avar = out_SVAR['Avar_est']
        avar = avar[np.logical_not(np.isnan(avar))]
        avar = np.reshape(avar, [np.size(b), np.size(b)])
        avar_this = avar[idx, idx]

        t_this = T * np.divide(np.power(b[idx], 2), avar_this)
        t_all[idx] = t_this
        t_all_p[idx] = 1 - scipy.stats.chi2.cdf(t_all[idx], 1)
        # cdf_this = scipy.stats.norm.cdf(t_this)
        # if cdf_this < 0.5:
        #     t_all_p[idx] = 2 * (scipy.stats.norm.cdf(t_this))
        # else:
        #     t_all_p[idx] = 2*(1-scipy.stats.norm.cdf(t_this))
    out_SVAR['t_all'] = t_all
    out_SVAR['t_all_p'] = t_all_p


    # Parameter wald
    out_SVAR['wald_all'], out_SVAR['wald_all_p'] = SVAR.SVARutilGMM.wald_param_all(B_est, options['restrictions'], out_SVAR['Avar_est'], T)

    # Fval - Rec
    out_SVAR['wald_rec'], out_SVAR['wald_rec_p'] = SVAR.SVARutilGMM.waldRec(B_est, out_SVAR['Avar_est'],
                                                                            options['restrictions'], T)


    if options['printOutput']:
            SVAR.estOutput.print_out(n, T, out_SVAR)

    return out_SVAR
