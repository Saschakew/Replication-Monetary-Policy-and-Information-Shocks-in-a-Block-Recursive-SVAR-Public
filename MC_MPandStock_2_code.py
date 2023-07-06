import os
import SVAR.MoG as MoG
import SVAR.estSVAR
import SVAR.estimatorGMM
import SVAR.SVARutil
import numpy as np
import pickle
import scipy

def OneMCIteration(path, jobno):
    np.random.seed(jobno)
    # SVAR Settings
    N = [5]
    T = [100,   250, 500, 1000, 5000]
    # shocks
    mu1, sigma1 = (-0.2, np.power(0.7, 2))
    mu2, sigma2 = (0.7501187648456057, np.power(1.4832737225521377, 2))
    lamb = 0.7895
    Omega = np.array([[mu1, sigma1], [mu2, sigma2]])

    mcit_out = dict()
    printOutput = False
    if False:
        mcit_out = dict()
        printOutput = True
        T_this= 300
        n= 5

    for T_this in T:
        for n in N:

            # Specitfy B_true
            V = np.array([[1, 0, 0, 0, 0],
                          [0.5, 1, 0, 0, 0],
                          [0.5, 0.5, 1, 0.5, 0],
                          [0.5, 0.5, 0.5, 1, 0],
                          [0., 0., 0., 0.75, 1]])
            b_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            O = SVAR.SVARutil.get_Orthogonal(b_true)
            B_true = np.matmul(V, O)

            B_true = B_true * 10

            # Draw structural shocks
            eps = np.empty([T_this, n])
            for i in range(n):
                if False:
                    df_t_i = 7  # + i*3
                    eps[:, i] = np.random.standard_t(df_t_i, size=T_this)
                    eps[:, i] = eps[:, i] / np.sqrt(df_t_i / (df_t_i - 2))  # normalize to unit variance
                else:
                    eps[:, i] = MoG.MoG_rnd(Omega, lamb, T_this)

            # Add heteroskedasticity
            if False:
                sigma_t = 1.5
                het_lambd = 0.5
                select = np.random.uniform(0, 1, T_this)
                eps[select > het_lambd, :] = sigma_t * eps[select > het_lambd, :]
                eps = eps / np.sqrt(np.power(het_lambd, 1) + np.power((1 - het_lambd), 1) * np.power(sigma_t,
                                                                                                     2))  # normalize to unit variance

            # Generate reduced form shocks
            u = np.matmul(B_true, np.transpose(eps))
            u = np.transpose(u)

            # General Output
            mcit_out['estimators'] = []
            mcit_out['T'] = T_this
            mcit_out['n'] = n
            mcit_out['B_true'] = B_true
            mcit_out['b_true'] = b_true

            try:
                estimator = 'CUE'
                prepOptions = dict()
                prepOptions['printOutput'] = printOutput
                prepOptions['addThirdMoments'] = True
                prepOptions['addFourthMoments'] = True
                prepOptions['bstartopt'] = 'specific'
                prepOptions['bstart'] = SVAR.get_BVector(B_true)
                prepOptions['Wpara'] = 'Independent'
                prepOptions['Avarparametric'] = 'Independent'
                prepOptions['Wstartopt'] = 'WoptBstart'
                prepOptions['moments_blocks'] = False
                prepOptions['S_func'] = True
                GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)
                # Find best permutation
                V = GMM_out['Avar_est']
                V = V[np.isnan(V) == False].reshape((np.sum(np.isnan(GMM_out['options']['restrictions'])),
                                                     np.sum(np.isnan(GMM_out['options']['restrictions']))))
                [Bbest, permutation] = SVAR.SVARutil.PermToB0(B_true, GMM_out['B_est'], V,
                                                              GMM_out['options']['restrictions'], T_this)
                # Refresh output for new B permutation
                GMM_out = SVAR.estOutput.genOutput(estimator, u, Bbest, GMM_out['options'])
                # Output estimator
                estimator_name = '00_CUE'
                mcit_out['estimators'].append(estimator_name)
                mcit_out[estimator_name] = dict()
                mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                mcit_out[estimator_name]['B_est'] = GMM_out['B_est']
                mcit_out[estimator_name]['loss'] = GMM_out['loss']
                mcit_out[estimator_name]['Sigma'] = GMM_out['Omega_all'][0]
                mcit_out[estimator_name]['avar'] = GMM_out['Avar_est']
                mcit_out[estimator_name]['wald_all'] = GMM_out['wald_all']
                mcit_out[estimator_name]['wald_all_p'] = GMM_out['wald_all_p']
                mcit_out[estimator_name]['wald_rec'] = GMM_out['wald_rec']
                mcit_out[estimator_name]['wald_rec_p'] = GMM_out['wald_rec_p']
                mcit_out[estimator_name]['J'] = GMM_out['J']
                mcit_out[estimator_name]['Jpvalue'] = GMM_out['Jpvalue']

                idxs = np.array([20, 21, 22, 23])
                mcit_out[estimator_name]['wald_0'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_0_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_1'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_1_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_2'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_2_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_3'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_3_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m1'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m1_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m2'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m2_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m3'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m3_p'] = np.zeros(np.size(idxs))
                for idx_i, idx in enumerate(idxs):
                    b = GMM_out['b_est'][idx]
                    avar = GMM_out['Avar_est']
                    avar = avar[np.logical_not(np.isnan(avar))]
                    avar = np.reshape(avar, [np.size(GMM_out['b_est']), np.size(GMM_out['b_est'])])
                    avar_this = avar[idx, idx]

                    t_this = T_this * np.divide(np.power(b, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_0'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_0_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b - 1, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_1'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_1_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b - 2, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_2'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_2_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b - 3, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_3'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_3_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b + 1, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_m1'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_m1_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b + 2, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_m2'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_m2_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b + 3, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_m3'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_m3_p'][idx_i] = t_p
            except:
                print("Error: " + estimator_name)

            # Blocks
            block1 = np.array([1, 2])
            block2 = np.array([3, 4])
            block3 = np.array([5, 5])
            blocks = list()
            blocks.append(block1)
            blocks.append(block2)
            blocks.append(block3)

            try:
                estimator = 'CUE'
                prepOptions = dict()
                prepOptions['printOutput'] = printOutput
                prepOptions['addThirdMoments'] = True
                prepOptions['addFourthMoments'] = True
                prepOptions['bstartopt'] = 'specific'
                prepOptions['bstart'] = SVAR.get_BVector(B_true,
                                                         restrictions=SVAR.SVARutil.getRestrictions_blocks(blocks))
                prepOptions['Wpara'] = 'Independent'
                prepOptions['Avarparametric'] = 'Independent'
                prepOptions['Wstartopt'] = 'WoptBstart'
                prepOptions['moments_blocks'] = True
                prepOptions['S_func'] = True
                prepOptions['blocks'] = blocks
                GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)
                # Find best permutation
                V = GMM_out['Avar_est']
                V = V[np.isnan(V) == False].reshape((np.sum(np.isnan(GMM_out['options']['restrictions'])),
                                                     np.sum(np.isnan(GMM_out['options']['restrictions']))))
                [Bbest, permutation] = SVAR.SVARutil.PermToB0(B_true, GMM_out['B_est'], V,
                                                              GMM_out['options']['restrictions'], T_this)
                # Refresh output for new B permutation
                GMM_out = SVAR.estOutput.genOutput(estimator, u, Bbest, GMM_out['options'])

                # Output estimator
                estimator_name = '01_CUE'
                mcit_out['estimators'].append(estimator_name)
                mcit_out[estimator_name] = dict()
                mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                mcit_out[estimator_name]['B_est'] = GMM_out['B_est']
                mcit_out[estimator_name]['loss'] = GMM_out['loss']
                mcit_out[estimator_name]['Sigma'] = GMM_out['Omega_all'][0]
                mcit_out[estimator_name]['avar'] = GMM_out['Avar_est']
                mcit_out[estimator_name]['wald_all'] = GMM_out['wald_all']
                mcit_out[estimator_name]['wald_all_p'] = GMM_out['wald_all_p']
                mcit_out[estimator_name]['wald_rec'] = GMM_out['wald_rec']
                mcit_out[estimator_name]['wald_rec_p'] = GMM_out['wald_rec_p']
                mcit_out[estimator_name]['J'] = GMM_out['J']
                mcit_out[estimator_name]['Jpvalue'] = GMM_out['Jpvalue']

                idxs = np.array([12, 13, 14, 15])
                mcit_out[estimator_name]['wald_0'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_0_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_1'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_1_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_2'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_2_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_3'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_3_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m1'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m1_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m2'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m2_p'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m3'] = np.zeros(np.size(idxs))
                mcit_out[estimator_name]['wald_m3_p'] = np.zeros(np.size(idxs))
                for idx_i, idx in enumerate(idxs):
                    b = GMM_out['b_est'][idx]
                    avar = GMM_out['Avar_est']
                    avar = avar[np.logical_not(np.isnan(avar))]
                    avar = np.reshape(avar, [np.size(GMM_out['b_est']), np.size(GMM_out['b_est'])])
                    avar_this = avar[idx, idx]

                    t_this = T_this * np.divide(np.power(b, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_0'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_0_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b - 1, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_1'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_1_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b - 2, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_2'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_2_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b - 3, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_3'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_3_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b + 1, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_m1'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_m1_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b + 2, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_m2'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_m2_p'][idx_i] = t_p

                    t_this = T_this * np.divide(np.power(b + 3, 2), avar_this)
                    t_p = 1 - scipy.stats.chi2.cdf(t_this, 1)
                    mcit_out[estimator_name]['wald_m3'][idx_i] = t_this
                    mcit_out[estimator_name]['wald_m3_p'][idx_i] = t_p
            except:
                print("Error: " + estimator_name)

            # Save results
            file_name = path + "/MCjobno_" + str(jobno) + "_n_" + str(n) + "_T_" + str(
                T_this) + ".data"
            with open(file_name, 'wb') as filehandle:
                pickle.dump(mcit_out, filehandle)
                # print('saved: ', file_name)

    return mcit_out
