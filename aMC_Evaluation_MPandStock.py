import pickle
import numpy as np
import pandas as pd
import os
import SVAR.SVARutil
pd.set_option('display.max_rows', 70)
pd.set_option('display.max_columns', 70)
pd.set_option('display.width', 150)




# ------------ Start: Specify version ------------#
useversion = 1 # select: 1 or 2
if useversion == 1:
    version = "MC_MPandStock_1"
    # Replicates Table 1 where estimator 00_CUE corresponds to the "SVAR-GMM" column and
    # estimator 01_CUE corresponds to the "block-recursive SVAR-GMM" column
    Table1Latex = ' '

    # Replicates Appendix Table 1 where estimator 00_GMMWF corresponds to the "SVAR-GMMWF" column and
    # estimator 01_GMMWF corresponds to the "block-recursive SVAR-GMMWF" column
    Table1AppendixLatex = ' '

    # Replicates Appendix Table 2 where estimator 00_PML corresponds to the "SVAR-PML" column and
    # estimator 01_PML corresponds to the "block-recursive SVAR-PML" column
    Table2AppendixLatex = ' '

if useversion == 2:
    version = "MC_MPandStock_2"
    # Replicate Table 2 where estimator 00_CUE corresponds to the "SVAR-GMM" column and
    # estimator 01_CUE corresponds to the "block-recursive SVAR-GMM" column
    Table2Latex = ' '

# ------------ End: Specify version ------------#


if version == "MC_MPandStock_1":
    Table1Latex +=  "\\begin{tabular}{ c|     c   c   }\n"
    Table1Latex += "T  &  SVAR-GMM & block-recursive SVAR-GMM \\\\\\hline\n"

    Table1AppendixLatex +=  "\\begin{tabular}{ c|     c   c   }\n"
    Table1AppendixLatex += "T  &  SVAR-GMMWF & block-recursive SVAR-GMMWF \\\\\\hline\n"

    Table2AppendixLatex +=  "\\begin{tabular}{ c|     c   c   }\n"
    Table2AppendixLatex += "T  &  SVAR-PML & block-recursive SVAR-PML \\\\\\hline\n"

if version == "MC_MPandStock_2":
    Table2Latex += "\\begin{tabular}{ c|     c c c c |  c c c c    }\n"
    Table2Latex += "&\\multicolumn{4}{c|}{SVAR-GMM }&\\multicolumn{4}{c}{block-recursive SVAR-GMM } \\\\\n"
    Table2Latex += "&$\\beta_1$ &$\\beta_2$ &$\\beta_3$ &$\\beta_4$  &$\\beta_1$ &$\\beta_2$ &$\\beta_3$ &$\\beta_4$\\\\\\hline\n"


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


def get_latex(this_std, this_bias):
    n = int(np.sqrt(np.shape(this_std)))
    counter = 0
    latex_string = '$\\begin{bmatrix}'
    for i in range(n):
        for j in range(n):
            latex_string += '\\underset{(' + str(this_std[counter]) + ')}{' + str(this_bias[counter]) + '}'
            if j < n - 1:
                latex_string += ' & '
            counter += 1
        latex_string += ' \\\\ '

    latex_string += '\\end{bmatrix}$ '
    return latex_string


N = np.unique(df.n)
T = np.unique(df["T"])
for n in N:
    for t in T:
        df_this = df[df.n == n]
        df_this = df_this[df_this["T"] == t]
        numMC_n = np.size(df_this[['n']])

        if numMC_n != 0:
            B_true = df_this.B_true[df_this.index[0]]
            b_true = SVAR.SVARutil.get_BVector(B_true)


            # Get estimators
            estimators = np.unique([df['estimators']])
            try:
                if type(estimators[0]) == list:
                    allsize = np.zeros(np.size(estimators), dtype=int)
                    for estidxthis, estimators_this in enumerate(estimators):
                        allsize[estidxthis] = np.size(estimators_this)
                    estimators = estimators[np.argmax(allsize)]
            except:
                pass


            bias = dict()
            wald0ect = dict()
            wald0val = dict()
            countEstimators = dict()

            for idx_estimator, estimator in enumerate(estimators):
                bias[estimator] = np.full([df_this.shape[0], np.size(b_true)] ,np.nan)
                countEstimators[estimator] = 0
                wald0ect[estimator] = np.full([df_this.shape[0],4], np.nan)
                wald0val[estimator] = np.full([df_this.shape[0]], np.nan)



            k = 0
            for index, row in df_this.iterrows():
                for estimator in row['estimators']:
                    countEstimators[estimator] += 1
                    B_est = row[estimator]['B_est']
                    bias[estimator][k, :] = SVAR.SVARutil.get_BVector(B_est) - b_true
                    try:
                        wald0ect[estimator][k] = row[estimator]['wald_0_p']
                        wald0val[estimator][k] = row[estimator]['wald_0']
                    except:
                        pass
                k += 1


            print("")
            # Print output
            print("For n=", n, ", T=", t, " and ", numMC_n, " simulations")
            print("T:", t)
            print("saved")
            print(countEstimators)

            out = pd.DataFrame()
            colnames = []
            # Generate output table
            counter = 0
            for estimator in estimators:
                bias_this = bias[estimator]

                out_this = [np.round(np.nanmean(bias_this, axis=0), 3)]   + b_true
                out = pd.concat([out, pd.DataFrame(out_this)], ignore_index=True)
                this_name = estimator + '_mean'
                colnames.append(this_name)
                counter += 1

                out_this = [np.nanmean(np.power(bias_this,2),axis=0)]
                out = pd.concat([out, pd.DataFrame(out_this)], ignore_index=True)
                this_name = estimator + '_var'
                colnames.append(this_name)
                counter += 1

            out = out.T
            colSums = np.sum(np.abs(out), axis=0)
            colSumsDat = pd.DataFrame([colSums], columns = out.columns)
            colSumsDat.rename(index={0: 'Sum abs'}, inplace=True)
            out = pd.concat([out, colSumsDat])

            # Label rows and coloums
            counter = 0
            for i in range(n):
                for j in range(n):
                    this_string = 'b(' + str(i + 1) + ',' + str(j + 1) + ')'
                    out.rename(index={counter: this_string}, inplace=True)
                    counter += 1
            out.columns = colnames
            cols = out.columns.to_list()

            print(' ')
            #print(out)

            if version == "MC_MPandStock_1":
                Table1Latex += '$ ' + str(t) + '$ &'
                Table1AppendixLatex += '$ ' + str(t) + '$ &'
                Table2AppendixLatex += '$ ' + str(t) + '$ &'
            if version == "MC_MPandStock_2":
                Table2Latex += '$ ' + str(t) + '$  &'

            for estimator in estimators:
                print('estimator: ', estimator)
                est_string_std = estimator + '_var'
                est_string_bias = estimator + '_mean'
                this_std = out[est_string_std]
                this_std = np.round(this_std, 2)
                this_bias = out[est_string_bias]
                this_bias = np.round(this_bias, 2)
                this_string = get_latex(this_std, this_bias)

                if version == "MC_MPandStock_1":
                    print(this_string)

                    if estimator == '00_CUE':
                        Table1Latex += this_string + ' & '
                    if estimator == '01_CUE':
                        Table1Latex += this_string    + "\\\\\n"


                    if estimator == '00_GMMWF':
                        Table1AppendixLatex += this_string + ' & '
                    if estimator == '01_GMMWF':
                        Table1AppendixLatex += this_string    + "\\\\\n"


                    if estimator == '00_PML':
                        Table2AppendixLatex += this_string + ' & '
                    if estimator == '01_PML':
                        Table2AppendixLatex += this_string    + "\\\\\n"

                if version == "MC_MPandStock_2":
                    print('J   / Wald 0 / Wald 5 - rejection rate at 10% ')
                    thiswald = np.round(np.sum(wald0ect[estimator] < 0.1,axis=0) / np.shape(wald0ect[estimator])[0]* 100,2)
                    print(thiswald)

                    # Convert the array elements to strings
                    thiswald_str = ['${}$'.format(x) for x in thiswald]
                    if estimator == '00_CUE':
                        # Join the elements with '&' to form the row
                        row_str = ' & '.join( thiswald_str) + ' & '
                        Table2Latex += row_str
                    if estimator == '01_CUE':
                        row_str = ' & '.join( thiswald_str) + '\\\\\n'
                        Table2Latex += row_str





            print(' ')
if version == "MC_MPandStock_1":
    Table1Latex += "\\end{tabular}"
    Table1AppendixLatex += "\\end{tabular}"
    Table2AppendixLatex += "\\end{tabular}"

    print("Table1Latex")
    print(Table1Latex)
    print("Table1AppendixLatex")
    print(Table1AppendixLatex)
    print("Table2AppendixLatex")
    print(Table2AppendixLatex)

    filename =  'FiguresAndTables\Table1Latex.txt'
    with open(filename, "w") as file:
        file.write(Table1Latex)


    filename =  'FiguresAndTables\TableAppendixB6Latex.txt'
    with open(filename, "w") as file:
        file.write(Table1AppendixLatex)


    filename =  'FiguresAndTables\TableAppendixB7Latex.txt'
    with open(filename, "w") as file:
        file.write(Table2AppendixLatex)


if version == "MC_MPandStock_2":
    Table2Latex += "\\end{tabular}"

    print("Table2Latex")
    print(Table2Latex)


    filename =  'FiguresAndTables\Table2Latex.txt'
    with open(filename, "w") as file:
        file.write(Table2Latex)