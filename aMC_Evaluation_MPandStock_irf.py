import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 70)
pd.set_option('display.max_columns', 70)
pd.set_option('display.width', 150)


# ------------ Start: Specify version ------------#
useversion = 2 # select: 1 or 2
if useversion == 1:
    T = np.array([100])
    # Replicates Appendix Figure 1

if useversion == 2:
    T = np.array([250])
    # Replicates Appendix Figure 2
# ------------ End: Specify version ------------#





version = "MC_MPandStock_3"


# Load all files of version
path = os.path.join("MCResults", version)
# Load collected data or create empty Data file
path_VersionData = os.path.join(path, str(version + ".data"))
try:
    with open(path_VersionData, 'rb') as filehandle:
        # read the data as binary data stream
        df = pickle.load(filehandle)
except:
    print("Run MC_collect_data.py")



N = np.unique(df.n)

UseEstimators = np.array(['00_CUE', '01_CUE'])
UseEstimatorsColors = np.array(['blue', 'orange'])


for n in N:
    for t in T:
        df_this = df[df.n == n]
        df_this = df_this[df_this["T"] == t]
        numMC_n = np.size(df_this[['n']])

        if numMC_n != 0:
            irf_true = df_this.irf_true[df_this.index[0]]



            # # Calculate Bias at each simulation
            estimators = np.unique([df['estimators']])
            try:
                if type(estimators[0]) == list:
                    allsize = np.zeros(np.size(estimators), dtype=int)
                    for estidxthis, estimators_this in enumerate(estimators):
                        allsize[estidxthis] = np.size(estimators_this)
                    estimators = estimators[np.argmax(allsize)]
            except:
                pass



            irf = dict()
            for idx_estimator, estimator in enumerate(estimators):
                irf[estimator] = np.full([df_this.shape[0], np.shape(irf_true)[0],np.shape(irf_true)[1],np.shape(irf_true)[2]] ,np.nan)


            k = 0
            for index, row in df_this.iterrows():
                for estimator in row['estimators']:
                    irf[estimator][k,:,:,:] = row[estimator]['irf']
                k += 1

            fig, ax = plt.subplots(4, 4, sharex=True, tight_layout=True)
            for idxestimator, estimator in enumerate(UseEstimators):
                irf_mean = np.mean(irf[estimator],axis=0)
                irf_median = np.median(irf[estimator],axis=0)
                alpha=0.1
                irf_qupper = np.quantile(irf[estimator],alpha/2,axis=0)
                irf_qlower = np.quantile(irf[estimator],1-alpha/2,axis=0)

                for i in range(4):
                    for j in range(4):
                        ylab = r'$shock: \epsilon_{' + str(i+1) + '}$'
                        xlab = r'$response: y_{' + str(j+1) + '}$'
                        if j == 0:
                            ax[i,j].set(ylabel=ylab)
                        if i==3:
                            ax[i,j].set(xlabel=xlab)

                        #ax[i,j].plot(irf_mean[:,i,j],color=UseEstimatorsColors[idxestimator])
                        ax[i,j].plot(irf_qupper[:,i,j],'--',color=UseEstimatorsColors[idxestimator])
                        ax[i,j].plot(irf_qlower[:,i,j],'--',color=UseEstimatorsColors[idxestimator])
                        ax[i,j].plot(irf_true[:,i,j],'*' ,color='red')
            if useversion == 1:
                fig_name = 'FiguresAndTables\Figure1Appendix.eps'
            if useversion == 2:
                fig_name = 'FiguresAndTables\Figure2Appendix.eps'

            fig.savefig(fig_name, format='eps', dpi=1200)
            plt.show()

