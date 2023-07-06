import numpy as np
import pandas as pd
from tabulate import tabulate
import SVAR

def genOutput(estimator, u, Bbest, options):
    if options['whiten']:
        Bbest = np.matmul(np.linalg.inv(options['V']), Bbest)
    est_SVAR = dict()
    est_SVAR['x'] = SVAR.get_BVector(Bbest, options['restrictions'],
                                     whiten=options['whiten'],
                                     blocks=options['blocks'])
    if estimator == 'GMM' or estimator == 'CUE' or estimator == 'CSUE':
        GMM_out = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)
    elif estimator == 'GMM_W':
        GMM_out = SVAR.estimatorGMMW.SVARout(est_SVAR, options, u)
    elif estimator == 'GMM_WF':
        GMM_out = SVAR.estimatorGMMWF.SVARout(est_SVAR, options, u)
    elif estimator == 'PML':
        GMM_out = SVAR.estimatorPML.SVARout(est_SVAR, options, u)
    else:
        print("Unknown estimator in genOutput")

    return GMM_out

def idx_Best(n):
    idx = np.array([])
    for i in np.arange(n):
        string = "B(" + str(i + 1) + ',:)'
        idx = np.append(idx, string)
    return idx


def print_Avar(V):
    n = int(np.sqrt(np.shape(V)[0]))
    print(pd.DataFrame(data=np.reshape(np.diag(V), [n, n]),
                       index=idx_Best(n),
                       columns=col_Best(  n)))


def print_Moments(omega):
    n = np.shape(omega)[0]
    omega = np.round(omega,2)
    omega = omega[:,1:-2]
    print(pd.DataFrame(data=omega, index=idx_moments(n),
                       columns=np.array([  'E[e^2]', 'E[e^3]', 'E[e^4]' ])))


def print_B(B):
    n = np.shape(B)[0]
    print(pd.DataFrame(data=B, index=idx_Best(n),
                       columns=col_Best(n)))


def idx_moments(n):
    idx = np.array([])
    for i in np.arange(n):
        string = "e" + str(i + 1)
        idx = np.append(idx, string)
    return idx


def col_Best(n):
    idx = np.array([])
    for i in np.arange(n):
        string = "B(:," + str(i + 1) + ')'
        idx = np.append(idx, string)
    return idx


def print_out(n,T,out_SVAR):
    estimator = out_SVAR['options']['estimator']
    loss = np.round(out_SVAR['loss'], 2)

    try:
        WUpdateSteps = out_SVAR['options']['kstep']
        if estimator == 'GMMcont':
            WUpdateSteps = 'CUE'
    except:
        WUpdateSteps = '-'


    try:
        scale = str()
        for i in range(n):
            scale = scale + str(np.round( np.diag(out_SVAR['options']['scale'])[i], 2))
            if i != n - 1:
                scale = scale + ' / '
    except:
        scale = '-'

    try:
        West = out_SVAR['options']['Wpara']
    except:
        West = '-'

    try:
        Avarest = out_SVAR['options']['Avarparametric']
    except:
        Avarest = '-'

    try:
        JTest = np.round(out_SVAR['J'], 2)
        JTestp = np.round(out_SVAR['Jpvalue'], 2)
    except:
        JTest = 'nan'
        JTestp = 'nan'
    try:
        Waldrec = np.round(out_SVAR['wald_rec'], 2)
        Waldrecp = np.round(out_SVAR['wald_rec_p'], 2)
    except:
        Waldrec = 'nan'
        Waldrecp = 'nan'
    try:
        moments_num = np.shape(out_SVAR['options']['moments'])[0]
        moments_num2 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 2)
        moments_num3 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 3)
        moments_num4 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 4)
    except:
        moments_num = 'nan'
        moments_num2 = 'nan'
        moments_num3 = 'nan'
        moments_num4 = 'nan'
    try:
        restrictions = out_SVAR['options']['restrictions']
        restrictions_num = np.power(n, 2) - np.sum(np.isnan(restrictions))
    except:
        restrictions = 'nan'
        restrictions_num = 'nan'

    try:
        omega = out_SVAR['omega']
        Ee2 = str()
        Ee3 = str()
        Ee4 = str()
        for i in range(n):
            Ee2 = Ee2 + str(np.round(omega[i, 1], 2))
            Ee3 = Ee3 + str(np.round(omega[i, 2], 2))
            Ee4 = Ee4 + str(np.round(omega[i, 3], 2))
            if i != n - 1:
                Ee2 = Ee2 + ' / '
                Ee3 = Ee3 + ' / '
                Ee4 = Ee4 + ' / '
    except:
        Ee2 = 'nan'
        Ee3 = 'nan'
        Ee4 = 'nan'

    try:
        moments_ident_num = np.shape(out_SVAR['options']['moments_ident'])[0]
        moments_ident_num2 = np.sum(np.sum(out_SVAR['options']['moments_ident'], axis=1) == 2)
        moments_ident_num3 = np.sum(np.sum(out_SVAR['options']['moments_ident'], axis=1) == 3)
        moments_ident_num4 = np.sum(np.sum(out_SVAR['options']['moments_ident'], axis=1) == 4)
    except:
        moments_ident_num = 'nan'
        moments_ident_num2 = 'nan'
        moments_ident_num3 = 'nan'
        moments_ident_num4 = 'nan'

    try:
        moments_overident_num = np.shape(out_SVAR['options']['moments_overident'])[0]
        moments_overident_num2 = np.sum(np.sum(out_SVAR['options']['moments_overident'], axis=1) == 2)
        moments_overident_num3 = np.sum(np.sum(out_SVAR['options']['moments_overident'], axis=1) == 3)
        moments_overident_num4 = np.sum(np.sum(out_SVAR['options']['moments_overident'], axis=1) == 4)
    except:
        moments_overident_num = 'nan'
        moments_overident_num2 = 'nan'
        moments_overident_num3 = 'nan'
        moments_overident_num4 = 'nan'

    try:
        moments_selected_num = np.shape(out_SVAR['options']['moments_overident'][out_SVAR['beta_est']==0])[0]
        moments_selected_num2 = np.sum(np.sum(out_SVAR['options']['moments_overident'][out_SVAR['beta_est']==0], axis=1) == 2)
        moments_selected_num3 = np.sum(np.sum(out_SVAR['options']['moments_overident'][out_SVAR['beta_est']==0], axis=1) == 3)
        moments_selected_num4 = np.sum(np.sum(out_SVAR['options']['moments_overident'][out_SVAR['beta_est']==0], axis=1) == 4)
    except:
        moments_selected_num = 'nan'
        moments_selected_num2 = 'nan'
        moments_selected_num3 = 'nan'
        moments_selected_num4 = 'nan'

    # TabData = list()
    # TabData.append([ estimator,  str(WUpdateSteps),   str(West) + '/' + str(Avarest) ,   str(BiasCorrection)  ])
    # tableSummary = tabulate(TabData, headers=np.array([
    #    str("Estimator:"   ), 'W update steps:', 'Estimator Wopt/Avar:',"Biascorrection:"]), tablefmt="github")
    # print(tableSummary)
    # print(" ")
    if estimator ==  'GMM':
        print('Estimator: GMM (steps='+ str(WUpdateSteps) + ')')
    elif estimator == 'CUE':
        print('Estimator: CUE')
    elif estimator == 'CSUE':
        print('Estimator: CSUE')
    elif estimator == 'GMM_W':
        print('Estimator: GMM white')
    elif estimator == 'GMM_WF':
        print('Estimator: GMM white fast')
    elif estimator == 'PML':
        print('Estimator: PML')
    elif estimator == 'LassoM':
        print('Estimator: LassoGMM (lambda=' +str(out_SVAR['options']['lambd']) + ')' )
    elif estimator == 'LassoMcont':
        print('Estimator: LassoCUE (lambda=' +str(out_SVAR['options']['lambd']) + ')' )
    elif estimator == 'LassoB':
        print('Estimator: LassoBGMM (lambda=' + str(out_SVAR['options']['lambd']) + ')')
    else:
        print("Unknown estimator")

    print('Estimator Wopt/Avar: ' + str(West) + '/' + str(Avarest))

    if estimator == 'LassoM' or estimator == 'LassoMcont':
        TabData = list()
        TabData.append(["#second: " + str(moments_ident_num2) + " /  " + str(moments_overident_num2) + " /  " + str(
            moments_selected_num2)])
        TabData.append(["#third: " + str(moments_ident_num3) + " /  " + str(moments_overident_num3) + " /  " + str(
            moments_selected_num3)])
        TabData.append(["#fourth: " + str(moments_ident_num4) + " /  " + str(moments_overident_num4) + " /  " + str(
            moments_selected_num4)])

        tableSummary = tabulate(TabData, headers=np.array([
            'Ident. / Overident. / Selected']), tablefmt="github")
        print(tableSummary)
        print(" ")

    # if estimator == 'GMM' or estimator == 'GMMcont' or estimator == 'GMM_W' or estimator == 'GMM_WF' or estimator == 'PML':
    TabData = list()
    TabData.append(["T=" + str(T), "#second: " + str(moments_num2), "WaldRec=" + str(Waldrec), "m=2: " + Ee2])
    TabData.append(["n=" + str(n), "#third: " + str(moments_num3), "WaldRec-pval=" + str(Waldrecp), "m=3: " + Ee3])
    TabData.append(["#restrictions:" + str(restrictions_num), "#fourth: " + str(moments_num4), "J=" + str(JTest),
                    "m=4: " + Ee4])
    TabData.append(
        ["#unknowns:" + str(np.power(n, 2) - restrictions_num), "->loss: " + str(loss), "J-pval=" + str(JTestp),
         " "  ])
    tableSummary = tabulate(TabData, headers=np.array([
       str("SVAR " ), 'Moments', 'Tests',"E[e_1^m] / ... / E[e_n^m]  "]), tablefmt="github")
    print(tableSummary)
    print(" ")


    # print("Moments of estimated structural shocks")
    # print_Moments(out_SVAR['omega'])
    # print(" ")

    TabData = list()
    b_est = np.round(SVAR.get_BVector(out_SVAR['B_est'], restrictions=restrictions, whiten=False), 2)

    try:
        Avar_est = np.diag(np.round(out_SVAR['Avar_est'], 2))
    except:
        Avar_est = np.full(np.shape(b_est), 'nan' )
    try:
        wald_all = np.round(out_SVAR['wald_all'], 2)
        wald_all_p = np.round(out_SVAR['wald_all_p'], 2)
    except:
        wald_all = np.full(np.shape(b_est), 'nan' )
        wald_all_p = np.full(np.shape(b_est), 'nan' )
    try:
        t_all = np.round(out_SVAR['t_all'], 2)
        t_all_p = np.round(out_SVAR['t_all_p'], 2)
    except:
        t_all = np.full(np.shape(b_est), 'nan' )
        t_all_p = np.full(np.shape(b_est), 'nan' )

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                b_est = np.insert(b_est, elementcounter, np.full(1, np.NaN), 0)
                wald_all = np.insert(wald_all, elementcounter, np.full(1, np.NaN), 0)
                wald_all_p = np.insert(wald_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all_p = np.insert(t_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all = np.insert(t_all, elementcounter, np.full(1, np.NaN), 0)
            elementcounter += 1
    rows = idx_Best(n)
    for i in range(n):
        this_row = list([rows[i]])
        for j in range(n):
            if np.isnan(b_est[n * i + j]):
                this_entry = '*' + str(np.round(restrictions[i, j],2)) + '*'
            else:
                this_entry = str(b_est[n * i + j])
            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list([str("(avar/wald/pval)")])
        for j in range(n):
            this_entry = " (" + str(Avar_est[n * i + j]) + '/' + str(wald_all[n * i + j]) + '/' + str(
                wald_all_p[n * i + j]) + ')'
                         # '/' + str(t_all_p[n * i + j]) + \

            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list()
        TabData.append(this_row)
    tableB = tabulate(TabData, headers=col_Best(n), stralign="center", tablefmt="github")
    print(tableB)
    print("Note: Restricted values are denoted by *x*")


def print_out_scaled(n,T,out_SVAR):
    estimator = out_SVAR['options']['estimator']+'_scaled'
    loss = np.round(out_SVAR['loss_scaled'], 2)
    try:
        JTest = np.round(out_SVAR['J_scaled'], 2)
        JTestp = np.round(out_SVAR['Jpvalue_scaled'], 2)
    except:
        JTest = 'nan'
        JTestp = 'nan'
    try:
        Waldrec = np.round(out_SVAR['wald_rec_scaled'], 2)
        Waldrecp = np.round(out_SVAR['wald_rec_p_scaled'], 2)
    except:
        Waldrec = 'nan'
        Waldrecp = 'nan'
    try:
        moments_num = np.shape(out_SVAR['options']['moments'])[0]
        moments_num2 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 2)
        moments_num3 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 3)
        moments_num4 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 4)
    except:
        moments_num = 'nan'
        moments_num2 = 'nan'
        moments_num3 = 'nan'
        moments_num4 = 'nan'
    try:
        restrictions = out_SVAR['options']['restrictions']
        restrictions_num = np.power(n, 2) - np.sum(np.isnan(restrictions))
    except:
        restrictions = 'nan'
        restrictions_num = 'nan'

    try:
        omega = out_SVAR['omega_scaled']
        Ee2 = str()
        Ee3 = str()
        Ee4 = str()
        for i in range(n):
            Ee2 = Ee2 + str(np.round(omega[i, 1], 2))
            Ee3 = Ee3 + str(np.round(omega[i, 2], 2))
            Ee4 = Ee4 + str(np.round(omega[i, 3], 2))
            if i != n - 1:
                Ee2 = Ee2 + ' / '
                Ee3 = Ee3 + ' / '
                Ee4 = Ee4 + ' / '
    except:
        Ee2 = 'nan'
        Ee3 = 'nan'
        Ee4 = 'nan'

    TabData = list()
    TabData.append(["T=" + str(T), "#second: " + str(moments_num2), "WaldRec=" + str(Waldrec) , "Notation: E[e_1^m] / ... / E[e_n^m]  "  ])
    TabData.append(["n=" + str(n), "#third: " + str(moments_num3), "WaldRec-pval=" + str(Waldrecp) , "m=2: " + Ee2])
    TabData.append(["#restrictions=" + str(restrictions_num), "#fourth: " + str(moments_num4), "J=" + str(JTest) , "m=3: " + Ee3])
    TabData.append(["#unknowns=" + str(np.power(n,2)-restrictions_num), "->loss: " + str(loss), "J-pval=" + str(JTestp), "m=4: " + Ee4])
    tableSummary = tabulate(TabData, headers=np.array([
       str("Estimator: " + estimator), '#moments', 'Tests','Moments of estimated shocks']), tablefmt="github")
    print(tableSummary)
    print(" ")

    # print("Moments of estimated structural shocks")
    # print_Moments(out_SVAR['omega'])
    # print(" ")

    TabData = list()
    b_est = np.round(SVAR.get_BVector(out_SVAR['B_est_scaled'], restrictions=restrictions, whiten=False), 2)

    try:
        Avar_est = np.diag(np.round(out_SVAR['Avar_est_scaled'], 2))
    except:
        Avar_est = np.full(np.shape(b_est), 'nan' )
    try:
        wald_all = np.round(out_SVAR['wald_all_scaled'], 2)
        wald_all_p = np.round(out_SVAR['wald_all_p_scaled'], 2)
    except:
        wald_all = np.full(np.shape(b_est), 'nan' )
        wald_all_p = np.full(np.shape(b_est), 'nan' )
    try:
        t_all = np.round(out_SVAR['t_all_scaled'], 2)
        t_all_p = np.round(out_SVAR['t_all_p_scaled'], 2)
    except:
        t_all = np.full(np.shape(b_est), 'nan' )
        t_all_p = np.full(np.shape(b_est), 'nan' )

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                b_est = np.insert(b_est, elementcounter, np.full(1, np.NaN), 0)
                wald_all = np.insert(wald_all, elementcounter, np.full(1, np.NaN), 0)
                wald_all_p = np.insert(wald_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all_p = np.insert(t_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all = np.insert(t_all, elementcounter, np.full(1, np.NaN), 0)
            elementcounter += 1
    rows = idx_Best(n)
    for i in range(n):
        this_row = list([rows[i]])
        for j in range(n):
            if np.isnan(b_est[n * i + j]):
                this_entry = '*' + str(restrictions[i, j]) + '*'
            else:
                this_entry = str(b_est[n * i + j])
            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list([str("(avar/wald/pval)")])
        for j in range(n):
            this_entry = " (" + str(Avar_est[n * i + j]) + '/' + str(wald_all[n * i + j]) + '/' + str(
                wald_all_p[n * i + j]) + ')'
                         # '/' + str(t_all_p[n * i + j]) + \

            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list()
        TabData.append(this_row)
    tableB = tabulate(TabData, headers=col_Best(n), stralign="center", tablefmt="github")
    print(tableB)
    print("Note: Restricted values are denoted by *x*")


def print_out_scaled(n,T,out_SVAR):
    estimator = out_SVAR['options']['estimator']+'_scaled'
    loss = np.round(out_SVAR['loss_scaled'], 2)
    try:
        JTest = np.round(out_SVAR['J_scaled'], 2)
        JTestp = np.round(out_SVAR['Jpvalue_scaled'], 2)
    except:
        JTest = 'nan'
        JTestp = 'nan'
    try:
        Waldrec = np.round(out_SVAR['wald_rec_scaled'], 2)
        Waldrecp = np.round(out_SVAR['wald_rec_p_scaled'], 2)
    except:
        Waldrec = 'nan'
        Waldrecp = 'nan'
    try:
        moments_num = np.shape(out_SVAR['options']['moments'])[0]
        moments_num2 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 2)
        moments_num3 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 3)
        moments_num4 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 4)
    except:
        moments_num = 'nan'
        moments_num2 = 'nan'
        moments_num3 = 'nan'
        moments_num4 = 'nan'
    try:
        restrictions = out_SVAR['options']['restrictions']
        restrictions_num = np.power(n, 2) - np.sum(np.isnan(restrictions))
    except:
        restrictions = 'nan'
        restrictions_num = 'nan'

    try:
        omega = out_SVAR['omega_scaled']
        Ee2 = str()
        Ee3 = str()
        Ee4 = str()
        for i in range(n):
            Ee2 = Ee2 + str(np.round(omega[i, 1], 2))
            Ee3 = Ee3 + str(np.round(omega[i, 2], 2))
            Ee4 = Ee4 + str(np.round(omega[i, 3], 2))
            if i != n - 1:
                Ee2 = Ee2 + ' / '
                Ee3 = Ee3 + ' / '
                Ee4 = Ee4 + ' / '
    except:
        Ee2 = 'nan'
        Ee3 = 'nan'
        Ee4 = 'nan'

    TabData = list()
    TabData.append(["T=" + str(T), "#second: " + str(moments_num2), "WaldRec=" + str(Waldrec) , "Notation: E[e_1^m] / ... / E[e_n^m]  "  ])
    TabData.append(["n=" + str(n), "#third: " + str(moments_num3), "WaldRec-pval=" + str(Waldrecp) , "m=2: " + Ee2])
    TabData.append(["#restrictions=" + str(restrictions_num), "#fourth: " + str(moments_num4), "J=" + str(JTest) , "m=3: " + Ee3])
    TabData.append(["#unknowns=" + str(np.power(n,2)-restrictions_num), "->loss: " + str(loss), "J-pval=" + str(JTestp), "m=4: " + Ee4])
    tableSummary = tabulate(TabData, headers=np.array([
       str("Estimator: " + estimator), '#moments', 'Tests','Moments of estimated shocks']), tablefmt="github")
    print(tableSummary)
    print(" ")

    # print("Moments of estimated structural shocks")
    # print_Moments(out_SVAR['omega'])
    # print(" ")

    TabData = list()
    b_est = np.round(SVAR.get_BVector(out_SVAR['B_est_scaled'], restrictions=restrictions, whiten=False), 2)

    try:
        Avar_est = np.diag(np.round(out_SVAR['Avar_est_scaled'], 2))
    except:
        Avar_est = np.full(np.shape(b_est), 'nan' )
    try:
        wald_all = np.round(out_SVAR['wald_all_scaled'], 2)
        wald_all_p = np.round(out_SVAR['wald_all_p_scaled'], 2)
    except:
        wald_all = np.full(np.shape(b_est), 'nan' )
        wald_all_p = np.full(np.shape(b_est), 'nan' )
    try:
        t_all = np.round(out_SVAR['t_all_scaled'], 2)
        t_all_p = np.round(out_SVAR['t_all_p_scaled'], 2)
    except:
        t_all = np.full(np.shape(b_est), 'nan' )
        t_all_p = np.full(np.shape(b_est), 'nan' )

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                b_est = np.insert(b_est, elementcounter, np.full(1, np.NaN), 0)
                wald_all = np.insert(wald_all, elementcounter, np.full(1, np.NaN), 0)
                wald_all_p = np.insert(wald_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all_p = np.insert(t_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all = np.insert(t_all, elementcounter, np.full(1, np.NaN), 0)
            elementcounter += 1
    rows = idx_Best(n)
    for i in range(n):
        this_row = list([rows[i]])
        for j in range(n):
            if np.isnan(b_est[n * i + j]):
                this_entry = '*' + str(restrictions[i, j]) + '*'
            else:
                this_entry = str(b_est[n * i + j])
            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list([str("(avar/wald/pval)")])
        for j in range(n):
            this_entry = " (" + str(Avar_est[n * i + j]) + '/' + str(wald_all[n * i + j]) + '/' + str(
                wald_all_p[n * i + j]) + ')'
                         # '/' + str(t_all_p[n * i + j]) + \

            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list()
        TabData.append(this_row)
    tableB = tabulate(TabData, headers=col_Best(n), stralign="center", tablefmt="github")
    print(tableB)
    print("Note: Restricted values are denoted by *x*")


def print_out_scaled(n,T,out_SVAR):
    estimator = out_SVAR['options']['estimator']+'_scaled'
    loss = np.round(out_SVAR['loss_scaled'], 2)
    try:
        JTest = np.round(out_SVAR['J_scaled'], 2)
        JTestp = np.round(out_SVAR['Jpvalue_scaled'], 2)
    except:
        JTest = 'nan'
        JTestp = 'nan'
    try:
        Waldrec = np.round(out_SVAR['wald_rec_scaled'], 2)
        Waldrecp = np.round(out_SVAR['wald_rec_p_scaled'], 2)
    except:
        Waldrec = 'nan'
        Waldrecp = 'nan'
    try:
        moments_num = np.shape(out_SVAR['options']['moments'])[0]
        moments_num2 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 2)
        moments_num3 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 3)
        moments_num4 = np.sum(np.sum(out_SVAR['options']['moments'], axis=1) == 4)
    except:
        moments_num = 'nan'
        moments_num2 = 'nan'
        moments_num3 = 'nan'
        moments_num4 = 'nan'
    try:
        restrictions = out_SVAR['options']['restrictions']
        restrictions_num = np.power(n, 2) - np.sum(np.isnan(restrictions))
    except:
        restrictions = 'nan'
        restrictions_num = 'nan'

    try:
        omega = out_SVAR['omega_scaled']
        Ee2 = str()
        Ee3 = str()
        Ee4 = str()
        for i in range(n):
            Ee2 = Ee2 + str(np.round(omega[i, 1], 2))
            Ee3 = Ee3 + str(np.round(omega[i, 2], 2))
            Ee4 = Ee4 + str(np.round(omega[i, 3], 2))
            if i != n - 1:
                Ee2 = Ee2 + ' / '
                Ee3 = Ee3 + ' / '
                Ee4 = Ee4 + ' / '
    except:
        Ee2 = 'nan'
        Ee3 = 'nan'
        Ee4 = 'nan'

    TabData = list()
    TabData.append(["T=" + str(T), "#second: " + str(moments_num2), "WaldRec=" + str(Waldrec) , "Notation: E[e_1^m] / ... / E[e_n^m]  "  ])
    TabData.append(["n=" + str(n), "#third: " + str(moments_num3), "WaldRec-pval=" + str(Waldrecp) , "m=2: " + Ee2])
    TabData.append(["#restrictions=" + str(restrictions_num), "#fourth: " + str(moments_num4), "J=" + str(JTest) , "m=3: " + Ee3])
    TabData.append(["#unknowns=" + str(np.power(n,2)-restrictions_num), "->loss: " + str(loss), "J-pval=" + str(JTestp), "m=4: " + Ee4])
    tableSummary = tabulate(TabData, headers=np.array([
       str("Estimator: " + estimator), '#moments', 'Tests','Moments of estimated shocks']), tablefmt="github")
    print(tableSummary)
    print(" ")

    # print("Moments of estimated structural shocks")
    # print_Moments(out_SVAR['omega'])
    # print(" ")

    TabData = list()
    b_est = np.round(SVAR.get_BVector(out_SVAR['B_est_scaled'], restrictions=restrictions, whiten=False), 2)

    try:
        Avar_est = np.diag(np.round(out_SVAR['Avar_est_scaled'], 2))
    except:
        Avar_est = np.full(np.shape(b_est), 'nan' )
    try:
        wald_all = np.round(out_SVAR['wald_all_scaled'], 2)
        wald_all_p = np.round(out_SVAR['wald_all_p_scaled'], 2)
    except:
        wald_all = np.full(np.shape(b_est), 'nan' )
        wald_all_p = np.full(np.shape(b_est), 'nan' )
    try:
        t_all = np.round(out_SVAR['t_all_scaled'], 2)
        t_all_p = np.round(out_SVAR['t_all_p_scaled'], 2)
    except:
        t_all = np.full(np.shape(b_est), 'nan' )
        t_all_p = np.full(np.shape(b_est), 'nan' )

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                b_est = np.insert(b_est, elementcounter, np.full(1, np.NaN), 0)
                wald_all = np.insert(wald_all, elementcounter, np.full(1, np.NaN), 0)
                wald_all_p = np.insert(wald_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all_p = np.insert(t_all_p, elementcounter, np.full(1, np.NaN), 0)
                t_all = np.insert(t_all, elementcounter, np.full(1, np.NaN), 0)
            elementcounter += 1
    rows = idx_Best(n)
    for i in range(n):
        this_row = list([rows[i]])
        for j in range(n):
            if np.isnan(b_est[n * i + j]):
                this_entry = '*' + str(restrictions[i, j]) + '*'
            else:
                this_entry = str(b_est[n * i + j])
            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list([str("(avar/wald/pval)")])
        for j in range(n):
            this_entry = " (" + str(Avar_est[n * i + j]) + '/' + str(wald_all[n * i + j]) + '/' + str(
                wald_all_p[n * i + j]) + ')'
                         # '/' + str(t_all_p[n * i + j]) + \

            this_row.extend([this_entry])
        TabData.append(this_row)

        this_row = list()
        TabData.append(this_row)
    tableB = tabulate(TabData, headers=col_Best(n), stralign="center", tablefmt="github")
    print(tableB)
    print("Note: Restricted values are denoted by *x*")