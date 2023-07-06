import os
import SVAR.MoG as MoG
import SVAR.estSVAR
import SVAR.estimatorGMM
import SVAR.SVARutil
import numpy as np
import pickle

def OneMCIteration(path, jobno):
    np.random.seed(jobno)
    # SVAR Settings
    N = [4]
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

    for T_this in T:
        for n in N:

            # Specitfy B_true
            V = np.array([[1, 0, 0, 0], [0.5, 1, 0, 0], [0.5, 0.5, 1, 0.5], [0.5, 0.5, 0.5, 1]])
            b_true = [0, 0, 0, 0, 0, 0]
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

            AR1 = np.array([[0.5, 0.0, 0.0, 0.0],
                            [0.1, 0.5, 0.0, 0.0],
                            [0.1, 0.1, 0.5, 0.0],
                            [0.1, 0.1, 0.1, 0.5]])
            AR2 = np.array([[0., 0.0, 0.0, 0.0],
                            [0.0, 0., 0.0, 0.0],
                            [0.0, 0.0, 0., 0.0],
                            [0.0, 0.0, 0.0, 0.]])
            AR = np.dstack((AR1, AR2))
            const = np.zeros(n)
            trend = np.zeros(n)
            trend2 = np.zeros(n)
            ystart = np.zeros([4, n])
            y = SVAR.estSVARbootstrap.simulate_SVAR2(eps, B_true, AR, const, trend, trend2, ystart)

            # Estimate reduced form
            opt_redform = dict()
            opt_redform['add_const'] = False
            opt_redform['add_trend'] = False
            opt_redform['add_trend2'] = False
            out_redform = SVAR.OLS_ReducedForm(y, lags=1, **opt_redform)
            out_redform['u']
            out_redform['AR'][:, :, 0]

            # Generate reduced form shocks
            u = out_redform['u']

            # General Output
            mcit_out['estimators'] = []
            mcit_out['T'] = T_this
            mcit_out['n'] = n
            mcit_out['B_true'] = B_true
            mcit_out['b_true'] = b_true

            mcit_out['irf_true'] = SVAR.get_IRF(B_true, AR, irf_length=10)

            try:
                estimator = 'CUE'
                prepOptions = dict()
                prepOptions['printOutput'] = printOutput
                prepOptions['addThirdMoments'] = True
                prepOptions['addFourthMoments'] = True
                # prepOptions['bstartopt'] = 'CUEN'
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
                mcit_out[estimator_name]['irf'] = SVAR.get_IRF(GMM_out['B_est'], out_redform['AR'][:, :, 0],
                                                               irf_length=10)
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

            except:
                print("Error: " + estimator_name)

            try:
                estimator = 'PML'
                prepOptions = dict()
                prepOptions['printOutput'] = False
                GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)

                # Output estimator
                estimator_name = '00_PML'
                mcit_out['estimators'].append(estimator_name)
                mcit_out[estimator_name] = dict()
                mcit_out[estimator_name]['irf'] = SVAR.get_IRF(GMM_out['B_est'], out_redform['AR'][:, :, 0],
                                                               irf_length=10)
                mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                mcit_out[estimator_name]['B_est'] = GMM_out['B_est']
                mcit_out[estimator_name]['loss'] = GMM_out['loss']
                mcit_out[estimator_name]['Sigma'] = GMM_out['Omega_all'][0]
                mcit_out[estimator_name]['avar'] = np.nan
                mcit_out[estimator_name]['wald_all'] = np.nan
                mcit_out[estimator_name]['wald_all_p'] = np.nan
                mcit_out[estimator_name]['wald_rec'] = np.nan
                mcit_out[estimator_name]['wald_rec_p'] = np.nan
                mcit_out[estimator_name]['J'] = np.nan
                mcit_out[estimator_name]['Jpvalue'] = np.nan

            except:
                print("Error: " + estimator_name)

            try:
                estimator = 'GMM_WF'
                prepOptions = dict()
                prepOptions['printOutput'] = False
                prepOptions['addThirdMoments'] = True
                prepOptions['addFourthMoments'] = True
                prepOptions['Avarparametric'] = 'Independent'
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
                estimator_name = '00_GMMWF'
                mcit_out['estimators'].append(estimator_name)
                mcit_out[estimator_name] = dict()
                mcit_out[estimator_name]['irf'] = SVAR.get_IRF(GMM_out['B_est'], out_redform['AR'][:, :, 0],
                                                               irf_length=10)
                mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                mcit_out[estimator_name]['B_est'] = GMM_out['B_est']
                mcit_out[estimator_name]['loss'] = GMM_out['loss']
                mcit_out[estimator_name]['Sigma'] = GMM_out['Omega_all'][0]
                mcit_out[estimator_name]['avar'] = GMM_out['Avar_est']
                mcit_out[estimator_name]['wald_all'] = GMM_out['wald_all']
                mcit_out[estimator_name]['wald_all_p'] = GMM_out['wald_all_p']
                mcit_out[estimator_name]['wald_rec'] = GMM_out['wald_rec']
                mcit_out[estimator_name]['wald_rec_p'] = GMM_out['wald_rec_p']
                mcit_out[estimator_name]['J'] = np.nan
                mcit_out[estimator_name]['Jpvalue'] = np.nan



            except:
                print("Error: " + estimator_name)

            # Blocks
            block1 = np.array([1, 2])
            block2 = np.array([3, 4])
            blocks = list()
            blocks.append(block1)
            blocks.append(block2)

            try:
                estimator = 'CUE'
                prepOptions = dict()
                prepOptions['printOutput'] = False
                prepOptions['addThirdMoments'] = True
                prepOptions['addFourthMoments'] = True
                # prepOptions['bstartopt'] = 'CUEN'
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
                mcit_out[estimator_name]['irf'] = SVAR.get_IRF(GMM_out['B_est'], out_redform['AR'][:, :, 0],
                                                               irf_length=10)
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


            except:
                print("Error: " + estimator_name)

            try:
                estimator = 'PML'
                prepOptions = dict()
                prepOptions['printOutput'] = False
                prepOptions['blocks'] = blocks
                GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)
                # Output estimator
                estimator_name = '01_PML'
                mcit_out['estimators'].append(estimator_name)
                mcit_out[estimator_name] = dict()
                mcit_out[estimator_name]['irf'] = SVAR.get_IRF(GMM_out['B_est'], out_redform['AR'][:, :, 0],
                                                               irf_length=10)
                mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                mcit_out[estimator_name]['B_est'] = GMM_out['B_est']
                mcit_out[estimator_name]['loss'] = GMM_out['loss']
                mcit_out[estimator_name]['Sigma'] = GMM_out['Omega_all'][0]
                mcit_out[estimator_name]['avar'] = np.nan
                mcit_out[estimator_name]['wald_all'] = np.nan
                mcit_out[estimator_name]['wald_all_p'] = np.nan
                mcit_out[estimator_name]['wald_rec'] = np.nan
                mcit_out[estimator_name]['wald_rec_p'] = np.nan
                mcit_out[estimator_name]['J'] = np.nan
                mcit_out[estimator_name]['Jpvalue'] = np.nan
            except:
                print("Error: " + estimator_name)

            try:
                estimator = 'GMM_WF'
                prepOptions = dict()
                prepOptions['printOutput'] = False
                prepOptions['addThirdMoments'] = True
                prepOptions['addFourthMoments'] = True
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
                estimator_name = '01_GMMWF'
                mcit_out['estimators'].append(estimator_name)
                mcit_out[estimator_name] = dict()
                mcit_out[estimator_name]['irf'] = SVAR.get_IRF(GMM_out['B_est'], out_redform['AR'][:, :, 0],
                                                               irf_length=10)
                mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                mcit_out[estimator_name]['B_est'] = GMM_out['B_est']
                mcit_out[estimator_name]['loss'] = GMM_out['loss']
                mcit_out[estimator_name]['Sigma'] = GMM_out['Omega_all'][0]
                mcit_out[estimator_name]['avar'] = GMM_out['Avar_est']
                mcit_out[estimator_name]['wald_all'] = GMM_out['wald_all']
                mcit_out[estimator_name]['wald_all_p'] = GMM_out['wald_all_p']
                mcit_out[estimator_name]['wald_rec'] = GMM_out['wald_rec']
                mcit_out[estimator_name]['wald_rec_p'] = GMM_out['wald_rec_p']
                mcit_out[estimator_name]['J'] = np.nan
                mcit_out[estimator_name]['Jpvalue'] = np.nan


            except:
                print("Error: " + estimator_name)

            # Save results
            file_name = path + "/MCjobno_" + str(jobno) + "_n_" + str(n) + "_T_" + str(
                T_this) + ".data"
            with open(file_name, 'wb') as filehandle:
                pickle.dump(mcit_out, filehandle)
                # print('saved: ', file_name)

    return mcit_out
