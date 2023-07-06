import os
# avoid server overflow with parallelization
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import SVAR
import matplotlib.pyplot as plt

## Specify SVAR
name = "Application_MPandStock_HFupdate"


# Set bootstrap options
opt_bootstrap = dict()
opt_bootstrap['number_bootstrap'] = 1000
opt_bootstrap['path'] = "BootstrapIRF/" + str(name)
opt_bootstrap['parallel'] = True
opt_bootstrap['num_cores_boot'] = 5

# Create new version folder if it does not exist
path_bootstrapIRF = os.path.join(opt_bootstrap['path'])
Path(path_bootstrapIRF).mkdir(parents=True, exist_ok=True)
path = os.path.join("BootstrapIRF", name)



if __name__ == '__main__':
    np.random.seed(0)

    # Load Data
    # Load high-frequency data
    hfdata = pd.read_csv("data/MPandStock/hfdataUpdate.csv", sep=";")
    hfyear = hfdata["year"].to_numpy()
    hfmonth = hfdata["month"].to_numpy()
    hfday = hfdata["day"].to_numpy()
    hfff = hfdata["ff4_hf"].to_numpy()
    hfsp = hfdata["sp500_hf"].to_numpy()
    ONRUN2 = hfdata["ONRUN2"].to_numpy()


    # High-frequency SVAR model
    hfstartyear = 1994
    hfday = hfday[hfyear >= hfstartyear]
    hfmonth = hfmonth[hfyear >= hfstartyear]
    hfsp = hfsp[hfyear >= hfstartyear]
    hfff = hfff[hfyear >= hfstartyear]
    ONRUN2 = ONRUN2[hfyear >= hfstartyear]
    hfyear = hfyear[hfyear >= hfstartyear]

    # Create high-frequency data for SVAR
    hfu = np.array([hfff, ONRUN2, hfsp]).T
    hfu = hfu - np.mean(hfu, axis=0)
    hfu = hfu * 100

    print("Summary of high-frequency data:")
    SVAR.SVARbasics.summarize_shocks(hfu, varnames=np.array(['hfff', 'ONRUN2',  'hfsp' ]))

    # Specify high-frequency SVAR estimator
    estimator = 'CUE'
    prepOptions = dict()
    prepOptions['printOutput'] = True
    prepOptions['estimator'] = estimator
    prepOptions['printOutput'] = True
    prepOptions['addThirdMoments'] = True
    prepOptions['addFourthMoments'] = True
    prepOptions['bstartopt'] = 'GMM_WF'
    prepOptions['Avarparametric'] = 'Independent'
    prepOptions['Wpara'] = 'Independent'
    prepOptions['S_func'] = True

    # Estimate high-frequency SVAR
    print("High-frequency SVAR estimation:")
    SVAR_out = SVAR.SVARest(hfu, estimator=estimator, prepOptions=prepOptions)
    hf_Best = SVAR_out['B_est']

    print("Summary of high-frequency shocks:")
    TabData = SVAR.SVARbasics.summarize_shocks(SVAR_out['e'], varnames=np.array(['mp', 'fg',  'CBinfo' ]))


    # Save high-frequency shocks
    hf_MPtrad = SVAR_out['e'][:, 0]
    hf_MPfg = SVAR_out['e'][:, 1]
    hf_CBinfo = SVAR_out['e'][:, 2]
    filename = 'FiguresAndTables\hf_shocks_update.csv'
    # Create a list of lists representing the rows
    rows = [
        hfyear.tolist(),
        hfmonth.tolist(),
        hfday.tolist(),
        hf_CBinfo.tolist(),
        hf_MPtrad.tolist(),
        hf_MPfg.tolist()
    ]
    # Define the header row
    header = ["Year", "Month", "Day", "CBInfo", "MP traditional", "forward guidance"]
    # Write the data to the CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(header)
        # Write the data rows
        writer.writerows(zip(*rows))

    year = np.repeat(np.arange(1994, 2020), 12)
    month = np.tile(np.arange(1, 13), len(year) // 12)

    month = month[: -6]
    year = year[: -6]

    # Calculate proxy for CB information
    proxy_info_CBinfo = np.full(np.shape(year), 0.0)
    proxy_info_MPtrad = np.full(np.shape(year), 0.0)
    proxy_info_MPfg = np.full(np.shape(year), 0.0)
    for t in range(np.size(year)):
        thisyear = year[t]
        thismonth = month[t]
        proxy_info_CBinfo[t] = np.sum(hf_CBinfo[np.logical_and(hfyear == thisyear, hfmonth == thismonth)])
        proxy_info_MPtrad[t] = np.sum(hf_MPtrad[np.logical_and(hfyear == thisyear, hfmonth == thismonth)])
        proxy_info_MPfg[t] = np.sum(hf_MPfg[np.logical_and(hfyear == thisyear, hfmonth == thismonth)])


    # Save high-frequency proxy variables
    # Convert ndarray variables to lists
    filename = 'FiguresAndTables\hf_proxy_variables_update.csv'
    # Create a list of lists representing the rows
    rows = [
        year.tolist(),
        month.tolist(),
        proxy_info_CBinfo.tolist(),
        proxy_info_MPtrad.tolist(),
        proxy_info_MPfg.tolist()
    ]
    # Define the header row
    header = ["Year", "Month", "Proxy CBInfo", "Proxy MP traditional", "Proxy forward guidance"]
    # Write the data to the CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(header)
        # Write the data rows
        writer.writerows(zip(*rows))



    # Plotting the variables
    x = year + month / 12
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # Plotting the variables on respective subplots
    ax1.plot(x, proxy_info_CBinfo   , color='blue')
    ax2.plot(x, proxy_info_MPtrad , color='green')
    ax3.plot(x, proxy_info_MPfg , color='red')

    # Customize the subplots
    ax1.set_ylabel(r'$\epsilon_{t}^{CBinfo}$', fontsize=16, fontweight='bold')
    ax2.set_ylabel(r'$\epsilon_{t}^{mp}$', fontsize=16, fontweight='bold')
    ax3.set_ylabel(r'$\epsilon_{t}^{fg}$', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.suptitle('Monetary Policy Proxy Variables', fontsize=16, fontweight='bold')

    # Adjust layout and spacing
    plt.tight_layout()

    # Save the plot as a JPG image
    plt.savefig('FiguresAndTables\proxy_plot_updated.jpg', dpi=300)

    # Display the plot
    plt.show()