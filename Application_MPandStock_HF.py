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

## Specify SVAR
name = "Application_MPandStock_HF"


# Set bootstrap options
opt_bootstrap = dict()
opt_bootstrap['number_bootstrap'] = 1000
opt_bootstrap['path'] = "BootstrapIRF/" + str(name)
opt_bootstrap['parallel'] = True
opt_bootstrap['num_cores_boot'] = 10

# Create new version folder if it does not exist
path_bootstrapIRF = os.path.join(opt_bootstrap['path'])
Path(path_bootstrapIRF).mkdir(parents=True, exist_ok=True)
path = os.path.join("BootstrapIRF", name)



if __name__ == '__main__':
    np.random.seed(0)

    # Load Data
    # Load low-frequency data
    data = pd.read_csv("data/MPandStock/dataJK.csv", sep=";")
    year = data["year"].to_numpy()
    month = data["month"].to_numpy()
    # Load high-frequency data
    hfdata = pd.read_csv("data/MPandStock/hfdata.csv", sep=";")
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

    # Generate and save Table 3 in the appendix
    def GenerateAppendixTableC8(data):
        table = "\\begin{tabular}{ c| c c c c c }\n"
        table += "\t& $\\varepsilon_{t}^{mp}$ & $\\varepsilon^{fg}_t$ & $\\varepsilon^{CBinfo}_t$ \\\\\\hline\n"

        for row in data:
            table += "\t" + row[0] + " & $" + " $ & $".join(row[1:]) + " $ \\\\\n"

        table += "\\end{tabular}"

        return table
    TableC8 = GenerateAppendixTableC8([TabData[4],TabData[5],TabData[6]])
    print("TableC8")
    print(TableC8)
    filename =  'FiguresAndTables\TableC8Latex.txt'
    with open(filename, "w") as file:
        file.write(TableC8)

    # Save high-frequency shocks
    hf_MPtrad = SVAR_out['e'][:, 0]
    hf_MPfg = SVAR_out['e'][:, 1]
    hf_CBinfo = SVAR_out['e'][:, 2]
    filename = 'FiguresAndTables\hf_shocks.csv'
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




    # Calculate high-frequency bootstrap quantiles of B estimate
    hf_Bsave = np.zeros([3, 3, opt_bootstrap['number_bootstrap']])
    prepOptions['printOutput'] = False
    prepOptions['bstartopt'] = 'specific'
    prepOptions['bstart'] = SVAR_out['b_est']
    for i in range(opt_bootstrap['number_bootstrap'] ):
        hfu_T = np.shape(hfu)[0]
        hfu_resample_index = np.random.choice(hfu_T, hfu_T, replace=True)
        hfu_resample = hfu[hfu_resample_index, :]
        SVAR_out = SVAR.SVARest(hfu_resample, estimator=estimator, prepOptions=prepOptions)
        hf_Bsave[:, :, i] = SVAR_out['B_est']
    print("High-frequency B estimate: hf_Best")
    hf_Best = np.round(hf_Best, 2)
    print(hf_Best)
    print("High-frequency B upper quantile estimate: hf_Bsave_q32")
    hf_Bsave_q32 = np.round(np.quantile(hf_Bsave, 0.32 / 2, axis=2), 2)
    print(hf_Bsave_q32)
    print("High-frequency B upper quantile estimate: hf_Bsave_q68")
    hf_Bsave_q68 = np.round(np.quantile(hf_Bsave, 1 - 0.32 / 2, axis=2), 2)
    print(hf_Bsave_q68)


    def GenerateTable4(output):
        table = "\\begin{tabular}{l l | ccc}\n"
        table += "\t& & Shock: $\\epsilon_{t}^{mp}$ & Shock: $\\epsilon_{t}^{fg}$  & Shock: $\\epsilon_{t}^{CBinfo}$  \\\\\n"
        table += "\t\\hline\n"

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if j == 0:
                    table += "Response: &"
                    if i == 0:
                        table += f" $i^{{(3M)}}_{{t}}$ &"
                    elif i == 1:
                        table += f" $i^{{(2Y)}}_{{t}}$ &"
                    elif i == 2:
                        table += f" $s_{{t}}$ &"
                else:
                    table += "\t"

                val = output[0,i, j ]
                lower_quantile = output[1,i, j ]
                upper_quantile = output[2,i, j ]

                table += f"$\\underset{{({lower_quantile}/{upper_quantile})}}{{{val:.2f}}}$"

                if j < output.shape[1] - 1:
                    table += "&"
                else:
                    table += "\\\\\n"

        table += "\\end{tabular}"
        return table
    Table4Latex = GenerateTable4(np.array([hf_Best, hf_Bsave_q32, hf_Bsave_q68]))
    print("Table4Latex")
    print(Table4Latex)
    filename =  'FiguresAndTables\Table4Latex.txt'
    with open(filename, "w") as file:
        file.write(Table4Latex)

    # Save high-frequency proxy variables
    # Convert ndarray variables to lists
    year = year.tolist()
    month = month.tolist()
    proxy_info_CBinfo = proxy_info_CBinfo.tolist()
    proxy_info_MPtrad = proxy_info_MPtrad.tolist()
    proxy_info_MPfg = proxy_info_MPfg.tolist()
    filename = 'FiguresAndTables\hf_proxy_variables.csv'
    # Create a list of lists representing the rows
    rows = [
        year,
        month,
        proxy_info_CBinfo,
        proxy_info_MPtrad,
        proxy_info_MPfg
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
