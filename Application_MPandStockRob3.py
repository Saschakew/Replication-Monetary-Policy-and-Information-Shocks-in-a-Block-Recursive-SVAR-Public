import os
# avoid server overflow with parallelization
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import pandas as pd
import numpy as np
import SVAR
import subprocess

## Specify SVAR
name = "Application_MPandStock_m3"

# Specify SVAR estimatorestimator_CUE_Blocks = True
estimator_CUE_Blocks = True
prepOptions = dict()
estimator = 'CUE'
prepOptions['estimator'] = estimator
prepOptions['printOutput'] = True
prepOptions['addThirdMoments'] = True  # use all third moments
prepOptions['addFourthMoments'] = True  # use all fourth moments
prepOptions['moments_blocks'] = True
prepOptions['bstartopt'] = 'GMM_WF'
prepOptions['Avarparametric'] = 'Independent'
prepOptions['Wpara'] = 'Independent'
prepOptions['S_func'] = True

# Set empty reduced form SVAR options
opt_redform = dict()
opt_redform['add_const'] = True
opt_redform['add_trend'] = False
opt_redform['add_trend2'] = False
opt_redform['lags'] = 12
# Set bootstrap options
opt_bootstrap = dict()
opt_bootstrap['number_bootstrap'] = 1000
opt_bootstrap['path'] = "BootstrapIRF/" + str(name)
opt_bootstrap['parallel'] = True
opt_bootstrap['num_cores_boot'] = 4
# Set IRF options
opt_irf = dict()
opt_irf['irf_length'] = 60
sym=True # symmetric True/False
alphas =  np.array([  0.32 / 2])  # confidence bands




if __name__ == '__main__':
    np.random.seed(0)

    # Specify list of estimators
    estimators = {'estimator_CUE_Blocks': estimator_CUE_Blocks}

    # Create new version folder if it does not exist
    path_bootstrapIRF = os.path.join(opt_bootstrap['path'])
    Path(path_bootstrapIRF).mkdir(parents=True, exist_ok=True)
    path = os.path.join("BootstrapIRF", name)
    # Create new folders
    for estimator_name, value in estimators.items():
        if value:
            pathEstimator = os.path.join(path, estimator_name)
            pathEstimatorPlot = os.path.join(pathEstimator, "Plot")
            pathEstimatorIRFs = os.path.join(pathEstimator, "IRFs")
            pathEstimatorIRFFirst = os.path.join(pathEstimator, "IRFFirst")
            pathEstimatorEstimationFirst = os.path.join(pathEstimator, "EstimationFirst")
            pathEstimatorCV = os.path.join(pathEstimator, "CV")

            Path(pathEstimatorPlot).mkdir(parents=True, exist_ok=True)
            Path(pathEstimatorIRFs).mkdir(parents=True, exist_ok=True)
            Path(pathEstimatorIRFFirst).mkdir(parents=True, exist_ok=True)
            Path(pathEstimatorEstimationFirst).mkdir(parents=True, exist_ok=True)
            Path(pathEstimatorCV).mkdir(parents=True, exist_ok=True)

    ## DATA
    # Load low-frequency data
    data = pd.read_csv("data/MPandStock/dataJK.csv", sep=";")
    year = data["year"].to_numpy()
    month = data["month"].to_numpy()
    gs1 = data["gs1"].to_numpy()
    logsp500 = data["logsp500"].to_numpy()
    us_gdpdef = data["us_gdpdef"].to_numpy()
    us_rgdp = data["us_rgdp"].to_numpy()
    ebpnew = data["ebpnew"].to_numpy()
    us_ip = data["us_ip"].to_numpy()
    us_cpi = data["us_cpi"].to_numpy()

    ## Select model
    tempindex = np.full(np.shape(year), True)
    tempindex[year < 1984] = False
    tempindex[year > 2008] = False
    date = year
    data_block1 = np.array([us_rgdp[tempindex],us_gdpdef[tempindex]])
    data_block2 = np.array([logsp500[tempindex],
                                  gs1[tempindex]])
    cummulative = [False, False, False, False]
    varnames = np.array(['y', '\pi', 's', 'i'])
    shocknames = np.array(['y', '\pi', 'info', 'mp'])
    # SVAR Data
    data_SVAR = np.concatenate((data_block1, data_block2))
    data_SVAR = np.transpose(data_SVAR)
    T, n = np.shape(data_SVAR)


    print("Summary of low-frequency data:")
    SVAR.SVARbasics.summarize_shocks(data_SVAR, varnames=varnames)

    # Estimate reduced-form VAR
    out_redform = SVAR.OLS_ReducedForm(data_SVAR, **opt_redform)
    u = out_redform['u']
    AR = out_redform['AR']


    print('Summary of low-frequency reduced form shocks')
    SVAR.SVARbasics.summarize_shocks(u, varnames=varnames)




    ## Low-frequency SVAR estimation
    if estimator_CUE_Blocks:
        # Specify options of SVAR estimator
        estimator_name = "estimator_CUE_Blocks"

        # Set Block-Restrictions
        block1 = np.array([1, np.shape(data_block1)[0]])
        block2 = np.array([np.shape(data_block1)[0]+1, n])
        blocks = list()
        blocks.append(block1)
        blocks.append(block2)
        prepOptions['blocks'] = blocks


        # Estimate simulatenous interaction of SVAR
        print("Low-frequency SVAR estimation:")
        SVAR_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)


        print('Summary of low-frequency structural shocks:')
        SVAR.SVARbasics.summarize_shocks(SVAR_out['e'], varnames=varnames)


        # Bootstrap SVAR
        out_irf, out_bootstrap, out_svarB, out_redformAR, = SVAR.bootstrap_SVAR(data_SVAR, estimator=estimator,
                                                         options_bootstrap=opt_bootstrap,
                                                         options_redform=opt_redform, options_irf=opt_irf,
                                                         prepOptions=prepOptions,
                                                         estimatorname=estimator_name )
        out_boootstrap_irf, out_bootstrap_B, out_bootstrap_AR = SVAR.SVARbasics.transform_SVARbootstrap_out(out_bootstrap)


        # Plot IRFs with bootstrap bands
        fig = SVAR.SVARbasics.plot_IRF(out_irf, irf_bootstrap=out_boootstrap_irf, alphas=alphas, shocks=np.arange(2,4),
                                       responses=[], shocknames=shocknames,
                                       responsnames=varnames, cummulative=cummulative, sym=sym)
        # Generate and save Figure 4 in the appendix
        fig_name =  'FiguresAndTables\FigureAppendixC6.pdf'
        fig.savefig(fig_name, format='pdf', dpi=300)
        eps_filename =  'FiguresAndTables\FigureAppendixC6.eps'
        subprocess.run(['pdf2ps', fig_name, eps_filename])
        fig_name = opt_bootstrap['path'] + '/' + estimator_name + '/Plot/irf.png'
        fig.savefig(fig_name, format='png', dpi=300)
