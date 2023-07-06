import numpy as np
import SVAR.SVARbasics
import SVAR.estSVAR
import SVAR.estPrepare
from SVAR.estPrepare import prepare_bootstrapIRF
import pickle
import pathos
import copy
mp = pathos.helpers.mp


def bootstrap_estimation(y,  estimator='GMM', opt_bootstrap=dict(), opt_redform=dict(),
                         prepOptions=dict(),
                         opt_SVAR=dict(), opt_irf=dict(), prepared=False, Bfirst=[]):
    out_redform = SVAR.SVARbasics.OLS_ReducedForm(y, **opt_redform)



    # Estimate B
    if prepared:
        opt_SVAR['printOutput'] = False
        out_svar = SVAR.SVARest(out_redform['u'], estimator=estimator, options=opt_SVAR, prepared=True)
    else:
        prepOptions['printOutput'] = True
        out_svar = SVAR.SVARest(out_redform['u'], estimator=estimator, prepOptions=prepOptions)

    # Find permutation
    if opt_bootstrap['find_perm'] == True:
        if Bfirst != []:
            try:
                # Find best permutation
                T_this=np.shape(out_redform['u'])[0]
                V = out_svar['Avar_est']
                V = V[np.isnan(V) == False].reshape((np.sum(np.isnan(out_svar['options']['restrictions'])),
                                                     np.sum(np.isnan(out_svar['options']['restrictions']))))
                [Bbest, permutation] = SVAR.SVARutil.PermToB0(Bfirst, out_svar['B_est'], V, out_svar['options']['restrictions'],
                                                              T_this)
                est_SVAR = dict()

                if  out_svar['options']['whiten']:
                    Bbest = np.matmul(np.linalg.inv( out_svar['options']['V']), Bbest)

                est_SVAR['x'] = SVAR.get_BVector(Bbest, out_svar['options']['restrictions'], whiten=out_svar['options']['whiten'],blocks=out_svar['options']['blocks'])

                if estimator == 'CUE' or estimator == 'CSUE':
                    out_svar = SVAR.estimatorGMM.SVARout(est_SVAR, out_svar['options'], out_redform['u'])
                elif estimator == 'GMM_W':
                    out_svar = SVAR.estimatorGMMW.SVARout(est_SVAR, out_svar['options'], out_redform['u'])
                elif estimator == 'GMM_WF':
                    out_svar = SVAR.estimatorGMMWF.SVARout(est_SVAR, out_svar['options'], out_redform['u'])
                elif estimator == 'PML':
                    out_svar = SVAR.estimatorPML.SVARout(est_SVAR, out_svar['options'], out_redform['u'])
                else:
                    print("Bootstrap find best permutation unknown estimator")
            except:
                print("Boostrap could not find best permutation")



    # get IRF
    out_irf = SVAR.SVARbasics.get_IRF(out_svar['B_est'], out_redform['AR'],   **opt_irf)

    return out_redform, out_svar, out_irf, out_svar['options']


def bootstrap_SVAR(y, estimator='GMM',
                   options_bootstrap=dict(), options_redform=dict(), prepOptions=dict(), options_irf=dict(),
                    estimatorname = ''):
    # Bundle inputs to options
    options_bootstrap, options_redform, options_irf = prepare_bootstrapIRF(**options_bootstrap, **options_redform,
                                                                   **options_irf)

    parallel = options_bootstrap['parallel']
    num_cores_boot = options_bootstrap['num_cores_boot']
    seed_start = options_bootstrap['seed_start']
    saveout = options_bootstrap['saveout']


    out_redform, out_svar, out_irf, opt_SVAR = bootstrap_estimation(y,  estimator,
                                                                    opt_bootstrap=options_bootstrap,
                                                                    opt_redform=options_redform,
                                                                    prepOptions=prepOptions, opt_SVAR=[],
                                                                    opt_irf=options_irf,
                                                                    prepared=False )


    # save first Estimation
    if saveout:
        out_svar_save = copy.deepcopy(out_svar)
        out_svar_save['options']['Jacobian']=np.nan
        out_svar_save['options']['S_func'] = np.nan
        out_svar_save['options']['Sdel_func'] = np.nan
        file_name = options_bootstrap['path'] + "/" + estimatorname + "/EstimationFirst/" + str(
            seed_start) + ".data"
        with open(file_name, 'wb') as filehandle:

            pickle.dump([out_redform, out_svar_save], filehandle)

        # save first IRF
        file_name = options_bootstrap['path'] + "/" + estimatorname + "/IRFFirst/" + str(
            seed_start) + ".data"
        with open(file_name, 'wb') as filehandle:
            pickle.dump(out_irf, filehandle)




    if num_cores_boot > 1:
        parallel = True


    def bootstrap_iter(y, estimator, options_bootstrap, options_redform, prepOptions, opt_SVAR,
                       options_irf, out_redform, jobno=0, seed_start=0, estimatorname='', Bfirst=[]):


        y_sim = simulate_SVAR(y,**out_redform, jobno=jobno, seed_start=seed_start)

        out_redform_iter, out_svar_iter, out_irf_iter, _ = bootstrap_estimation(y_sim,  estimator,
                                                                                options_bootstrap,
                                                                                options_redform, prepOptions,
                                                                                opt_SVAR, options_irf,
                                                                                prepared=True,
                                                                                Bfirst=Bfirst)

        # Save results
        if saveout:
            file_name = options_bootstrap['path'] +"/" + estimatorname  +"/IRFs/Bootstrap_" + str(
                seed_start) + 'run_' + str(jobno) + ".data"
            with open(file_name, 'wb') as filehandle:
                pickle.dump(out_irf_iter, filehandle)

        return out_redform_iter['AR'], out_svar_iter['B_est'], out_irf_iter


    if parallel:
        poolBoot = mp.Pool(num_cores_boot)  # ProcessPool #mp.Pool(num_coresCV)
        result_boot_async = poolBoot.starmap_async(bootstrap_iter, [(y, estimator, options_bootstrap,
                                                                     options_redform, prepOptions, opt_SVAR,
                                                                     options_irf, out_redform, bootstrap_run,
                                                                     seed_start,
                                                                     estimatorname,out_svar['B_est'])
                                                                    for bootstrap_run in
                                                                    range(options_bootstrap['number_bootstrap'])])
        result_boot = result_boot_async.get()

        poolBoot.close()
        poolBoot.join()
        out_bootstrap = result_boot

    else:

        # Loop through iterations
        out_bootstrap = [bootstrap_iter(y, estimator, options_bootstrap,
                                            options_redform, prepOptions, opt_SVAR, options_irf, out_redform,
                                            bootstrap_run, seed_start, estimatorname,out_svar['B_est']) for bootstrap_run in
                             range(options_bootstrap['number_bootstrap'])]


    return out_irf, out_bootstrap, out_svar['B_est'], out_redform['AR'],




def simulate_SVAR(y,u, AR, const, trend, trend2, shuffle=True, jobno=1, seed_start=0):
    np.random.seed(seed_start + jobno)
    T, n = np.shape(u)
    lags = np.shape(AR)[2]

    if shuffle:
        u_resample_index = np.random.choice(T, T, replace=True)
        u_resample = u[u_resample_index, :]
    else:
        u_resample = u

    y_new = copy.deepcopy(y)
    for t in range(T):
        tmpsum = const + np.multiply(trend, t - lags) + np.multiply(trend2, t - lags)
        for j in range(lags):
            # if t - j > 0:
            tmpsum = tmpsum + np.matmul(AR[:, :, j], y_new[lags+t - j - 1])
            # else:
            #     tmpsum = tmpsum + np.matmul(AR[:, :, j], ystart[t-j-1])
        tmpsum = tmpsum + u_resample[t]
        y_new[lags+t, :] = tmpsum

    return y_new

def simulate_SVAR2(e,B, AR, const, trend, trend2,ystart):
    T, n = np.shape(e)
    lags = np.shape(AR)[2]



    y_new = np.zeros([T, n])
    for t in range(T):
        tmpsum = const + np.multiply(trend, t - lags) + np.multiply(trend2, t - lags)
        for j in range(lags):
            if t - j > 0:
                tmpsum = tmpsum + np.matmul(AR[:, :, j], y_new[t - j - 1])
            else:
                tmpsum = tmpsum + np.matmul(AR[:, :, j], ystart[t-j-1])
        tmpsum = tmpsum + np.matmul(B,e[t,:])
        y_new[t, :] = tmpsum

    return y_new