import numpy as np
import SVAR
from SVAR.SVARutilGMM import get_Moments,  get_S_uncentered,  get_g, get_G_Indep, get_S_Indep, get_S






def loss(u, b, W, restrictions, moments, moments_powerindex, blocks=False):
    whiten = True
    g = get_g(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks)
    Q = np.linalg.multi_dot([g, W, g])
    return Q

def gradient(u, b, Jacobian, W, restrictions, moments, moments_powerindex):
    e = SVAR.innovation(u, b, restrictions=restrictions,whiten=True)
    CoOmega = SVAR.SVARutil.get_CoOmega(e)

    Jac_temp = Jacobian(u=u, b=b, restrictions=restrictions, CoOmega=CoOmega)

    this_g = get_g(b=b[:], u=u[:], restrictions=restrictions[:],
                   moments=moments[:], moments_powerindex=moments_powerindex)

    dGMM_temp  =  2 * np.linalg.multi_dot([np.transpose(Jac_temp), W, this_g])
    return dGMM_temp

def get_W_fast(Moments):
    W = np.eye(np.shape(Moments)[0])

    counter = 0
    for moment in Moments:
        r = np.sum(moment)

        tmp = np.zeros(np.shape(moment))
        counter_in = 0
        for i in moment:
            tmp[counter_in] = np.math.factorial(moment[counter_in])
            counter_in += 1

        W[counter, counter] = np.math.factorial(r) / np.prod(tmp)
        counter += 1

    return W

def get_Avar(n, G2,GC, S22, S2C, SC2, SCC, W=[],restrictions=[]):
    Moments2 = SVAR.SVARutilGMM.get_Mr(2, n)
    Moments2 = np.append(Moments2, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)

    if np.array(W).size == 0:
        W = np.linalg.inv(SCC)

    if np.array(restrictions).size == 0:
        restrictions = np.full([n, n], np.nan)
    number_of_restrictions = np.sum(restrictions == 0)


    H11 = G2
    H12 = np.zeros([np.shape(Moments2)[0], np.shape(Moments2)[0]])
    H21 = np.matmul(np.matmul(np.transpose(GC), W), GC)
    H22 = - np.transpose(G2)
    H = np.hstack((np.vstack((H11, H21)), np.vstack((H12, H22))))
    H = np.linalg.inv(H)
    M2 = H[0:(n * n - number_of_restrictions), 0: np.shape(Moments2)[0]]
    MC_tmp1 = H[0: (n * n - number_of_restrictions), np.shape(Moments2)[0]:]
    MC_tmp2 = np.matmul(np.transpose(GC), W)
    MC = np.matmul(MC_tmp1, MC_tmp2)
    V1 = np.matmul(np.matmul(M2, S22), np.transpose(M2))
    V2 = np.matmul(np.matmul(M2, S2C), np.transpose(MC))
    V3 = np.matmul(np.matmul(MC, SC2), np.transpose(M2))
    V4 = np.matmul(np.matmul(MC, SCC), np.transpose(MC))
    V_est = V1 + V2 + V3 + V4

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                V_est = np.insert(V_est, elementcounter, np.full(np.shape(V_est)[1], np.NaN), 0)
                V_est = np.insert(V_est, elementcounter, np.full(np.shape(V_est)[0], np.NaN), 1)
            elementcounter += 1
    return V_est

def get_GMM_W_Avar_param(Moments, B, omega, restrictions=[], W=[]):
    # Calculates analytically the asymptotic variance of sqrt(T)(b_hat-b) with b_hat GMM estimator with optimal weighting

    n, n = np.shape(B)

    if np.array(restrictions).size == 0: restrictions = np.full([n, n], np.nan)
    number_of_restrictions = np.sum(restrictions == 0)

    Moments2 = SVAR.SVARutilGMM.get_Mr(2, n)
    Moments2 = np.append(Moments2, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)

    G2 = get_G_Indep(Moments2, B, omega, restrictions)
    GC = get_G_Indep(Moments, B, omega, restrictions)

    S22 = get_S_Indep(Moments2, Moments2, omega)
    S2C = get_S_Indep(Moments2, Moments, omega)
    SC2 = get_S_Indep(Moments, Moments2, omega)
    SCC = get_S_Indep(Moments, Moments, omega)

    if np.array(W).size == 0:  W = np.linalg.inv(SCC)

    H11 = G2
    H12 = np.zeros([np.shape(Moments2)[0], np.shape(Moments2)[0]])
    H21 = np.matmul(np.matmul(np.transpose(GC), W), GC)
    H22 = - np.transpose(G2)
    H = np.hstack((np.vstack((H11, H21)), np.vstack((H12, H22))))
    H = np.linalg.inv(H)
    M2 = H[0:(n * n - number_of_restrictions), 0: np.shape(Moments2)[0]]
    MC_tmp1 = H[0: (n * n - number_of_restrictions), np.shape(Moments2)[0]:]
    MC_tmp2 = np.matmul(np.transpose(GC), W)
    MC = np.matmul(MC_tmp1, MC_tmp2)
    V1 = np.matmul(np.matmul(M2, S22), np.transpose(M2))
    V2 = np.matmul(np.matmul(M2, S2C), np.transpose(MC))
    V3 = np.matmul(np.matmul(MC, SC2), np.transpose(M2))
    V4 = np.matmul(np.matmul(MC, SCC), np.transpose(MC))
    V = V1 + V2 + V3 + V4

    # np.linalg.det(np.matmul(np.transpose(G2),G2))
    # tmp=np.matmul(GC,np.transpose(GC))
    # test = np.linalg.inv(np.matmul(np.matmul(np.transpose(GC), tmp), GC))
    if False:
        H11 = np.matmul(np.matmul(np.transpose(GC), W), GC)
        H12 = - np.transpose(G2)
        H21 = G2
        H22 = np.zeros([np.shape(Moments2)[0], np.shape(Moments2)[0]])
        H = np.hstack((np.vstack((H11, H21)), np.vstack((H12, H22))))
        K = np.linalg.inv(H)
        K11 = K[:(n * n - number_of_restrictions),:(n * n - number_of_restrictions)]
        K12 = K[:(n * n - number_of_restrictions), (n * n - number_of_restrictions): ]
        K21 = K[(n * n - number_of_restrictions): ,:(n * n - number_of_restrictions)]
        K22 = K[(n * n - number_of_restrictions): , (n * n - number_of_restrictions):  ]


        tmp = np.linalg.inv(np.matmul(    np.transpose(G2) ,G2))
        tmp2 = - np.matmul( G2,tmp) # =K21
        - np.matmul(K21,np.matmul(    np.transpose(G2) ,G2)) #=G2


        tmp = np.linalg.inv(np.matmul(G2, np.transpose(G2)))
        tmp2 = - np.matmul( np.matmul( np.matmul(np.matmul(np.transpose(GC), W), GC), np.transpose(G2)), tmp)
        tmp3 = np.matmul(K21,tmp2 ) # = K22



        tmp =  np.matmul(np.matmul(np.matmul(np.transpose(GC), W), GC), np.transpose( np.matmul(np.matmul(np.transpose(GC), W), GC) ))
        np.linalg.det(tmp)



    # invGWG = np.linalg.inv(np.matmul(np.matmul(np.transpose(GC), W), GC))
    # invG2invGWGG2 = np.linalg.inv(np.matmul(np.matmul(G2, invGWG), np.transpose(G2)))
    # L1 = np.matmul(np.matmul(invGWG, np.transpose(G2)), invG2invGWGG2)
    # L2 = - np.matmul(L1, np.matmul(np.matmul(G2, invGWG), np.matmul(np.transpose(GC), W)))
    # L3 = np.matmul(invGWG, np.matmul(np.transpose(GC), W))
    # L4 = L2+L3
    # R1 = np.matmul(np.matmul(L1, S22), np.transpose(L1))
    # R2 = np.matmul(np.matmul(L1, S2C), np.transpose(L4))
    # R3 = np.matmul(np.matmul(L4, SC2), np.transpose(L1))
    # R4 = np.matmul(np.matmul(L4, SCC), np.transpose(L4))
    # R = R1 + R2 + R3 + R4


    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                V = np.insert(V, elementcounter, np.full(np.shape(V)[1], np.NaN), 0)
                V = np.insert(V, elementcounter, np.full(np.shape(V)[0], np.NaN), 1)
            elementcounter += 1

    return V



def prepareOptions(u,
                   addThirdMoments=True, addFourthMoments=True, moments=[], moments_blocks=True,   moments_MeanIndep=False,
                   bstart=[], bstartopt='Rec',
                   Avarparametric='Uncorrelated',
                   blocks=False, n_rec=False,
                   W=[], Wstartopt='I',
                   printOutput=True,
                    estimator = 'GMM_W'
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

    if Avarparametric == 'Uncorrelated' or Avarparametric == 'Independent' or Avarparametric == 'Uncorrelated_uncentered':
        options['Avarparametric'] = Avarparametric
    else:
        print("Invalid Avarparametric value. Set Avarparametric=Uncorrelated")
        options['Avarparametric'] = 'Uncorrelated'

    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions=[])
    options['restrictions'] = restrictions
    options['blocks'] = blocks

    moments = SVAR.estPrepare.prepare_moments('GMM_W', moments, addThirdMoments, addFourthMoments, moments_blocks,
                                              blocks, n, moments_MeanIndep)
    options['moments'] = moments
    options['moments_MeanIndep'] = moments_MeanIndep
    options['moments_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(moments)

    bstart = SVAR.estPrepare.prepare_bstart(estimator, bstart, u, options, bstartopt=bstartopt)
    options['bstart'] = bstart


    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=moments, restrictions=restrictions)
    options['Jacobian'] = Jacobian

    # Weighting
    if Wstartopt == 'I':
        W = np.eye(np.shape(moments)[0])
    elif Wstartopt == 'GMM_WF':
        W = get_W_fast(moments)
    elif Wstartopt == 'specific':
        W = W
    else:
        raise ValueError('Unknown option Wstartopt')
    options['W'] = W

    if np.shape(options['moments'])[0] < np.shape(options['bstart'])[0]:
        raise ValueError('Less moment conditions than parameters. The SVAR is not identified')

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
    CoOmega = SVAR.SVARutil.get_CoOmega(e)
    out_SVAR['omega'] = omega

    # out_SVAR['loss'] = est_SVAR['fun']
    z, options['V'] = SVAR.do_whitening(u, white=True)
    out_SVAR['loss'] = loss(z, b_est, options['W'], options['restrictions'], options['moments'], options['moments_powerindex'], blocks=options['blocks'])





    Moments2 = SVAR.SVARutilGMM.get_Mr(2, n)
    Moments2 = np.append(Moments2, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    if options['Avarparametric'] == 'Independent':
        G2 = get_G_Indep(Moments2, out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])
        GC = get_G_Indep(options['moments'], out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])

        S22 = get_S_Indep(Moments2, Moments2, out_SVAR['omega'])
        S2C = get_S_Indep(Moments2, options['moments'], out_SVAR['omega'])
        SC2 = get_S_Indep(options['moments'], Moments2, out_SVAR['omega'])
        SCC = get_S_Indep(options['moments'], options['moments'], out_SVAR['omega'])
    elif options['Avarparametric'] == 'Uncorrelated':
        Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=options['moments'], restrictions=options['restrictions'])
        Jacobian2 = SVAR.SVARutilGMM.generate_Jacobian_function(moments=Moments2, restrictions=options['restrictions'])

        GC = Jacobian(u=u, b= SVAR.get_BVector(B_est,restrictions=options['restrictions']) , restrictions=options['restrictions'], CoOmega=CoOmega)
        G2 = Jacobian2(u=u, b= SVAR.get_BVector(B_est,restrictions=options['restrictions']) , restrictions=options['restrictions'], CoOmega=CoOmega)

        if np.shape(options['moments'])[0] == 0:
            GC = get_G_Indep(options['moments'], out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])

        S = get_S(u, SVAR.get_BVector(B_est,restrictions=np.full([n, n], np.nan)),  np.append(Moments2, options['moments'], axis=0) , np.full([n, n], np.nan) )
        S22 = S[:np.shape(Moments2)[0],:np.shape(Moments2)[0]]
        S2C = S[:np.shape(Moments2)[0],np.shape(Moments2)[0]:]
        SC2 = S[np.shape(Moments2)[0]:,:np.shape(Moments2)[0]]
        SCC = S[np.shape(Moments2)[0]:,np.shape(Moments2)[0]:]
    elif options['Avarparametric'] == 'Uncorrelated_uncentered':
        Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=options['moments'],
                                                               restrictions=options['restrictions'])
        Jacobian2 = SVAR.SVARutilGMM.generate_Jacobian_function(moments=Moments2, restrictions=options['restrictions'])

        GC = Jacobian(u=u, b=SVAR.get_BVector(B_est, restrictions=options['restrictions']),
                      restrictions=options['restrictions'], CoOmega=CoOmega)
        G2 = Jacobian2(u=u, b=SVAR.get_BVector(B_est, restrictions=options['restrictions']),
                       restrictions=options['restrictions'], CoOmega=CoOmega)

        if np.shape(options['moments'])[0] == 0:
            GC = get_G_Indep(options['moments'], out_SVAR['B_est'], out_SVAR['omega'], options['restrictions'])
        S = get_S_uncentered(u, SVAR.get_BVector(B_est,restrictions=np.full([n, n], np.nan)),  np.append(Moments2, options['moments'], axis=0) , np.full([n, n], np.nan) )
        S22 = S[:np.shape(Moments2)[0],:np.shape(Moments2)[0]]
        S2C = S[:np.shape(Moments2)[0],np.shape(Moments2)[0]:]
        SC2 = S[np.shape(Moments2)[0]:,:np.shape(Moments2)[0]]
        SCC = S[np.shape(Moments2)[0]:,np.shape(Moments2)[0]:]

    V_est = get_Avar(n, G2, GC, S22, S2C, SC2, SCC, W=options['W'], restrictions=options['restrictions'])
    out_SVAR['Avar_est'] = V_est



    # V_est = get_GMM_W_Avar_param(options['moments'], B=out_SVAR['B_est'], omega=out_SVAR['omega'],
    #                                restrictions=options['restrictions'], W=options['W'])



    # Parameter wald
    out_SVAR['wald_all'], out_SVAR['wald_all_p'] = SVAR.SVARutilGMM.wald_param_all(B_est, options['restrictions'], out_SVAR['Avar_est'], T)

    # Fval - Rec
    out_SVAR['wald_rec'], out_SVAR['wald_rec_p'] = SVAR.SVARutilGMM.waldRec(B_est, out_SVAR['Avar_est'],
                                                                            options['restrictions'], T)


    if options['printOutput']:
            SVAR.estOutput.print_out(n, T, out_SVAR)


    return out_SVAR