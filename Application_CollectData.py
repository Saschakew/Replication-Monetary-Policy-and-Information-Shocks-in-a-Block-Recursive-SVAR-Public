import pickle
import pandas as pd
import os
import time
import numpy as np

def collectData_Application(estimatorname, model):
    path = 'BootstrapIRF/' + model + '/' + estimatorname + '/EstimationFirst'
    path_VersionData = os.path.join(path, "out_est.data")
    files = os.listdir(path)
    all_out_est = []
    for file in files:
        file

        if file.endswith(".data") and not file.startswith('out_est'):
            # path_this = os.path.join(path, file)
            path_this = path + '/' + file
            # print(path_this)
            with open(path_this, 'rb') as filehandle:
                # read the data as binary data stream
                placesList = pickle.load(filehandle)
                # store in Data
                all_out_est.append(placesList)

    AllFirstEstEqual = all((all_out_est[i][1]['B_est'] == all_out_est[0][1]['B_est']).all() for i in  np.arange(np.shape(all_out_est)[0]))
    if not (AllFirstEstEqual):
        print("Not all out_est are equal!")
    out_est = all_out_est[0]
    with open(path_VersionData, 'wb') as filehandle:
        pickle.dump(out_est, filehandle)



    path = 'BootstrapIRF/' + model + '/' + estimatorname + '/IRFFirst'
    path_VersionData = os.path.join(path,   "out_irf.data" )
    files = os.listdir(path)
    all_out_irf = []
    for file in files:
        file

        if file.endswith(".data") and not file.startswith('out_irf'):
            # path_this = os.path.join(path, file)
            path_this = path + '/' + file
            # print(path_this)
            with open(path_this, 'rb') as filehandle:
                # read the data as binary data stream
                placesList = pickle.load(filehandle)
                # store in Data
                all_out_irf.append(placesList)

    AllFirstIRFEqual = all((elem == all_out_irf[0]).all() for elem in all_out_irf)
    if not(AllFirstIRFEqual):
        print("Not all out_irf are equal!")
    out_irf = all_out_irf[0]
    with open(path_VersionData, 'wb') as filehandle:
        pickle.dump(out_irf, filehandle)



    # Specify path to MCResults
    path = 'BootstrapIRF/' + model + '/' + estimatorname + '/IRFs'

    # Load collected data or create empty Data file
    path_VersionData = os.path.join(path, str(estimatorname + ".data"))
    try:
        with open(path_VersionData, 'rb') as filehandle:
            # read the data as binary data stream
            Data = pickle.load(filehandle)
    except:
        Data = []
    # Load file names of collected data or create empty Data names file
    path_VersionNames = os.path.join(path, str(estimatorname + "_names.data"))
    try:
        with open(path_VersionNames, 'rb') as filehandle:
            # read the data as binary data stream
            Data_names = pickle.load(filehandle)
    except:
        Data_names = pd.DataFrame()

    # Load MC Data and add to data, data_names
    files = os.listdir(path)
    for file in files:
        if file.endswith(".data") and file.startswith("Bootstrap_") and not file.startswith(estimatorname):
            if file not in Data_names.values:
                #path_this = os.path.join(path, file)
                path_this = path + '/'+ file
                # print(path_this)
                with open(path_this, 'rb') as filehandle:
                    # read the data as binary data stream
                    try:
                        placesList = pickle.load(filehandle)
                        # store in Data
                        Data.append( placesList  )
                        Data_names = Data_names.append(pd.DataFrame([file]), ignore_index=True, sort=False)
                    except:
                        pass




    # Delete MC Data files
    for file in files:
        if file.endswith(".data") and file.startswith("Bootstrap_"):
            path_this = os.path.join(path, file)
            os.remove(path_this)

    # Save data, data_names
    with open(path_VersionData, 'wb') as filehandle:
        pickle.dump(Data, filehandle)
    print('saved Data ')
    with open(path_VersionNames, 'wb') as filehandle:
        pickle.dump(Data_names, filehandle)
    print('saved Data names ')





if __name__ == "__main__":
    # Specify version

    model = "Application_OilBRidge_M0_Bcenter_v2_5"
    estimatorname = 'estimator_WFGMMRest'
    # estimatorname='estimator_WFGMMRidge2'
    # estimatorname='estimator_WFGMM'

    # estimatorname = 'estimator_PML_Blocks'
    # estimatorname = 'estimator_WFGMM_Blocks'
    # estimatorname = 'estimator_GMMcont_Blocks_ND'
    # estimatorname = 'estimator_GMMcont_Blocks_N'
    # estimatorname = 'estimator_LGMMcont_Blocks'
    # estimatorname = 'estimator_LasscontCV_Blocks'
    # estimatorname = 'estimator_PML'
    # estimatorname = 'estimator_WFGMM'
    # estimatorname = 'estimator_GMMcont_ND'
    # estimatorname = 'estimator_GMMcont_N'
    # estimatorname='estimator_GMMcont_N_recursive'

    collectData_Application(estimatorname, model)

