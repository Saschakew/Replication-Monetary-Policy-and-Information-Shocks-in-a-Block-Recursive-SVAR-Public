import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import multiprocessing
import MC_MPandStock_2_code
from pathlib import Path
from aMC_CollectData import collectData
import numpy as np

np.random.seed(0)

# Allows to create new version folder if it does not exist
version = "MC_MPandStock_2"
path = os.path.join("MCResults", version)
Path(path).mkdir(parents=True, exist_ok=True)

runcollect = True

if __name__ == '__main__':
    num_cores = int(multiprocessing.cpu_count())


    # Set up the multiprocessing pool
    pool = multiprocessing.Pool(processes=num_cores)

    # Define the number of simulations to run and the number of processes to use
    num_simulations = 2000


    args = [(path, i) for i in range(num_simulations)]



    # Run the simulations in parallel
    results = pool.starmap_async(MC_MPandStock_2_code.OneMCIteration, args)

    pool.close()
    pool.join()


    # Collect Data
    if runcollect:
        collectData(version)

