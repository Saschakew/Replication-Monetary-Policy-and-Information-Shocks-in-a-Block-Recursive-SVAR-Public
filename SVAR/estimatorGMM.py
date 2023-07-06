import numpy as np
import SVAR
import SVAR.SVARutilGMM
from SVAR.SVARutilGMM import  get_S_uncentered, get_g, get_G_Indep, get_S_Indep, get_S
import scipy.stats


def loss(u, b, W, restrictions, moments, moments_powerindex, blocks=False ):
    whiten = False
    g = get_g(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks )
    Q = np.linalg.multi_dot([g, W, g])


    return Q


def gradient(u, b, Jacobian, W, restrictions, moments, moments_powerindex):
    e = SVAR.innovation(u, b, restrictions=restrictions)
    CoOmega = SVAR.SVARutil.get_CoOmega(e)

    Jac_temp = Jacobian(u=u, b=b, restrictions=restrictions, CoOmega=CoOmega)
    this_g = get_g(b=b[:], u=u[:], restrictions=restrictions[:],
                   moments=moments[:], moments_powerindex=moments_powerindex)
    dGMM_temp  =  2 * np.linalg.multi_dot([np.transpose(Jac_temp), W, this_g])
    return (dGMM_temp)


def gradient_cont(u, b, Jacobian, W, restrictions, moments, moments_powerindex, Sdel_func):
    n = np.shape(moments)[1]
    B = SVAR.get_BMatrix(b, restrictions=restrictions)
    A = np.linalg.inv(B)
    e = SVAR.innovation(u, b, restrictions=restrictions)
    omegaext = SVAR.SVARutil.get_omegaext(e)
    CoOmega = SVAR.SVARutil.get_CoOmega(e)

    Jac_temp = Jacobian(u=u, b=b, restrictions=restrictions, CoOmega=CoOmega)
    this_g = get_g(b=b[:], u=u[:], restrictions=restrictions[:],
                   moments=moments[:], moments_powerindex=moments_powerindex)
    dGMM_temp = 2 * np.linalg.multi_dot([np.transpose(Jac_temp), W, this_g])

    counter = 0
    for q in range(n):
        for l in range(n):
            # only if B element is not restricted
            if np.isnan(restrictions[q, l]):
                L = Sdel_func[counter](omegaext,A)
                dGMM_temp2 = np.linalg.multi_dot([np.transpose(this_g), W, L, W, this_g])
                dGMM_temp[counter] = dGMM_temp[counter] + dGMM_temp2
                counter = counter + 1

    return (dGMM_temp)


# Estimate Avar and tests
def get_Avar(n, G,S,W=[],restrictions=[]):
    if np.array(W).size == 0:
        W = np.linalg.inv(S)

    if np.array(restrictions).size == 0:
        restrictions = np.full([n, n], np.nan)

    M1 = np.linalg.inv(np.matmul(np.matmul(np.transpose(G), W), G))
    M2 = np.matmul(np.transpose(G), W)
    M = np.matmul(M1, M2)
    V_est = np.matmul(np.matmul(M, S), np.transpose(M))

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                V_est = np.insert(V_est, elementcounter, np.full(np.shape(V_est)[1], np.NaN), 0)
                V_est = np.insert(V_est, elementcounter, np.full(np.shape(V_est)[0], np.NaN), 1)
            elementcounter += 1
    return V_est




# Only for GMM (not w or wf)
def prepareOptions(u,
                   addThirdMoments=True, addFourthMoments=True, moments=[], moments_blocks=True,  moments_MeanIndep=False,
                   bstart=[], bstartopt='Rec',
                   Avarparametric='Uncorrelated',
                   restrictions=[], blocks=False, n_rec=False,
                   kstep=2, W=[], Wpara='Uncorrelated', Wstartopt='I',   S_func=False,
                   printOutput=True,
                     WupdateInOutput=False,
                   estimator='GMM'):
    options = dict()

    options['estimator'] = estimator

    T, n = np.shape(u)
    options['T'] = T
    options['n'] = n


    options['WupdateInOutput'] = WupdateInOutput

    options['printOutput'] = printOutput

    if Avarparametric == 'Uncorrelated' or Avarparametric == 'Uncorrelated_uncentered'  or Avarparametric == 'Independent':
        options['Avarparametric'] = Avarparametric
    else:
        print("Invalid Avarparametric value. Set Avarparametric=Uncorrelated")
        options['Avarparametric'] = 'Uncorrelated'

    options['whiten'] = False

    options['kstep'] = kstep

    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions)
    options['restrictions'] = restrictions
    options['blocks'] = blocks

    moments = SVAR.estPrepare.prepare_moments('GMM', moments, addThirdMoments, addFourthMoments, moments_blocks,
                                              blocks, n,moments_MeanIndep)
    options['moments'] = moments
    options['moments_MeanIndep'] = moments_MeanIndep
    options['moments_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(moments)

    options['moments_num'] = np.shape(options['moments'])[0]
    options['moments_num2']  = np.sum(np.sum(options['moments'], axis=1) == 2)
    options['moments_num3']  = np.sum(np.sum(options['moments'], axis=1) == 3)
    options['moments_num4']  = np.sum(np.sum(options['moments'], axis=1) == 4)


    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=moments, restrictions=restrictions)
    options['Jacobian'] = Jacobian



    Sdel_func = SVAR.SVARutilGMM.generat_S_del_functions(moments, restrictions)
    options['Sdel_func'] = Sdel_func

    bstart = SVAR.estPrepare.prepare_bstart('GMM', bstart, u, options, bstartopt=bstartopt )
    options['bstart'] = bstart

    if S_func:
        if Wpara == 'Independent':
            S_func = SVAR.SVARutilGMM.generate_SIndep_function(moments)
        elif Wpara == 'Uncorrolated':
            S_func = False
        elif Wpara == 'Uncorrelated_uncentered':
            S_func = False
    options['S_func'] = S_func

    # Weighting
    if Wpara == 'Uncorrelated' or  Wpara == 'Independent'  or Wpara == 'Uncorrelated_uncentered':
        options['Wpara'] = Wpara
    else:
        print("Invalid Wpara value. Set Wpara=Uncorrelated")
        Wpara = 'Uncorrelated'
        options['Wpara'] = 'Uncorrelated'
    W = SVAR.estPrepare.prepare_W(u, n, W, Wstartopt, moments, bstart, restrictions,   Wpara, options['S_func']  )
    options['W'] = W



    if np.shape(options['moments'])[0] < np.shape(options['bstart'])[0] :
        raise ValueError('Less moment conditions than parameters. The SVAR is not identified')

    return options


def SVARout(est_SVAR, options, u):
    T, n = np.shape(u)
    out_SVAR = dict()
    out_SVAR['options'] = options

    b_est = est_SVAR['x']
    out_SVAR['b_est'] = b_est


    B_est = SVAR.get_BMatrix(b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                             blocks=options['blocks'] )
    out_SVAR['B_est'] = B_est



    e = SVAR.innovation(u, b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                        blocks=options['blocks'] )
    out_SVAR['e'] = e

    g = get_g(b=b_est, u=u, restrictions=options['restrictions'], moments=options['moments'],
              moments_powerindex=SVAR.SVARutilGMM.get_Moments_powerindex(options['moments']),
              whiten=options['whiten'], blocks=options['blocks'] )
    out_SVAR['g'] = g


    Omega_all = SVAR.SVARutil.get_Omega(e)
    out_SVAR['Omega_all'] = Omega_all
    omega = SVAR.SVARutil.get_Omega_Moments(e)
    CoOmega = SVAR.SVARutil.get_CoOmega(e)
    out_SVAR['omega'] = omega




    if options['Avarparametric'] == 'Independent':
        G = get_G_Indep(options['moments'], out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])
        S = get_S_Indep(options['moments'], options['moments'], omega=out_SVAR['omega'])
    elif options['Avarparametric'] == 'Uncorrelated':
        G = options['Jacobian'](u=u, b=out_SVAR['b_est'], restrictions=options['restrictions'], CoOmega=CoOmega)
        S = get_S(u, out_SVAR['b_est'],   options['moments'], options['restrictions'] )
    elif options['Avarparametric'] == 'Uncorrelated_uncentered':
        G = options['Jacobian'](u=u, b=out_SVAR['b_est'], restrictions=options['restrictions'], CoOmega=CoOmega)
        S = get_S_uncentered(u, out_SVAR['b_est'],   options['moments'], options['restrictions'] )



    if options['WupdateInOutput']:
        options['W'] = np.linalg.inv(S)

    V_est = get_Avar(n, G, S, W=options['W'], restrictions=options['restrictions'])
    out_SVAR['G'] = G
    out_SVAR['S'] = S
    out_SVAR['Avar_est'] = V_est


    loss = SVAR.estimatorGMM.loss(u, SVAR.get_BVector(out_SVAR['B_est'],  restrictions=options['restrictions']),
                                                      restrictions=options['restrictions'],
                                                      moments=options['moments'],
                                                      moments_powerindex=options['moments_powerindex'],
                                                      W=options['W'] )
    out_SVAR['loss'] = loss

    out_SVAR['J'] = np.multiply(T, loss)
    out_SVAR['Jpvalue'] = 1 - scipy.stats.chi2.cdf(out_SVAR['J'], np.size(g) - np.size(b_est))


    t_all = np.empty_like(b_est)
    t_all_p = np.empty_like(b_est)
    for idx, b in np.ndenumerate(b_est):
        avar = out_SVAR['Avar_est']
        avar = avar[np.logical_not(np.isnan(avar) ) ]
        avar = np.reshape(avar,[np.size(b_est),np.size(b_est)])
        avar_this = avar[idx,idx]

        t_this = T*np.divide(np.power(b_est[idx],2),avar_this)
        t_all[idx] = t_this
        t_all_p[idx] = 1 - scipy.stats.chi2.cdf(t_all[idx], 1)
    out_SVAR['t_all'] = t_all
    out_SVAR['t_all_p'] = t_all_p

    # Parameter wald
    out_SVAR['wald_all'], out_SVAR['wald_all_p'] = SVAR.SVARutilGMM.wald_param_all(B_est, options['restrictions'], out_SVAR['Avar_est'], T)

    # Fval - Rec
    out_SVAR['wald_rec'], out_SVAR['wald_rec_p']  = SVAR.SVARutilGMM.waldRec(B_est, out_SVAR['Avar_est'], options['restrictions'], T)

    if options['printOutput']:
            SVAR.estOutput.print_out(n, T, out_SVAR)


    return out_SVAR
