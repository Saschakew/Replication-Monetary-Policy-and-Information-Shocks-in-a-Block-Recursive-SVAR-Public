import pickle
import pandas as pd
import os
import time

def collectData(version):
    # Specify path to MCResults
    path = os.path.join("MCResults", version)

    # Load collected data or create empty Data file
    path_VersionData = os.path.join(path, str(version + ".data"))
    if os.path.exists(path_VersionData):
        with open(path_VersionData, 'rb') as filehandle:
            Data = pickle.load(filehandle)
    else:
        Data = pd.DataFrame()

    # Load file names of collected data or create empty Data names file
    path_VersionNames = os.path.join(path, str(version + "_names.data"))
    if os.path.exists(path_VersionNames):
        with open(path_VersionNames, 'rb') as filehandle:
            Data_names = pickle.load(filehandle)
    else:
        Data_names = pd.DataFrame()

    # Load MC Data and add to data, data_names
    files = [file for file in os.listdir(path) if
             file.endswith(".data") and file.startswith("MCjobno") and not file.startswith(version)]
    new_files = [file for file in files if ~Data_names.isin([file]).any().any()]

    for file in new_files:
        path_this = os.path.join(path, file)
        with open(path_this, 'rb') as filehandle:
            try:
                placesList = pickle.load(filehandle)
                # Append data and filename to existing Data and Data_names
                Data = pd.concat([Data, pd.DataFrame([placesList])], ignore_index=True, sort=False)
                Data_names = pd.concat([Data_names, pd.DataFrame([file])], ignore_index=True, sort=False)
            except:
                pass


    # Delete MC Data files
    for file in files:
        if file.endswith(".data") and file.startswith("MCjobno"):
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
    version = "MC_GMM_4skewnorm"
    collectData(version)

    # for i in range(10):
    #     time.sleep(250)
    #     collectData(version)
