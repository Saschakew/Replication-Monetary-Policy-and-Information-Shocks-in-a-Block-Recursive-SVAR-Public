import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy
from scipy.stats import norm
import SVAR
from tabulate import tabulate
import scipy.stats


# def var_statsmodel(y,lags):
#     model = VAR(y)
#     results = model.fit(lags)
#     print(results.summary())
#     return results
#
# def var_infocrit(y,maxlags,crit='aic'):
#     model = VAR(y)
#     model.select_order(maxlags)
#     results = model.fit(maxlags=maxlags, ic=crit)
#     print(results.summary())
#     return results


def get_rhs_flex(y, this_lags, maxLag, add_const=True, add_trend=False, add_trend2=False):
    # LagsM = np.array([2, 3])

    T, n = np.shape(y)
    Time = np.arange(T - maxLag)

    if add_const:
        rhs = np.ones([1,T - maxLag])
    # else:
    #     rhs = np.zeros([T - maxLag])

    if add_trend:
        rhs = np.vstack((rhs, Time))
    # else:
    #     rhs = np.vstack((rhs, np.zeros([T - maxLag])))

    if add_trend2:
        rhs = np.vstack((rhs, np.power(Time, 2)))
    # else:
    #     rhs = np.vstack((rhs, np.zeros([T - maxLag])))
    try:
        rhs = np.transpose(rhs)
        for lag in range(maxLag):
            if lag < this_lags:
                rhs = np.append(rhs, y[(maxLag - lag - 1):(-lag - 1), :], axis=1)
            else:
                rhs = np.append(rhs, np.zeros([T - maxLag, n]), axis=1)
    except:
        for lag in range(maxLag):
            if lag == 0:
                if lag < this_lags:
                    rhs =   y[(maxLag - lag - 1):(-lag - 1), :]
                else:
                    rhs =  np.zeros([T - maxLag, n])
            else:
                if lag < this_lags:
                    rhs = np.append(rhs, y[(maxLag - lag - 1):(-lag - 1), :], axis=1)
                else:
                    rhs = np.append(rhs, np.zeros([T - maxLag, n]), axis=1)

    if "rhs" not in locals():
        rhs = []

    return rhs


def get_lhs(y, lags):
    lhs = y[lags:, :]
    return lhs


def get_ARMAtrices(coefmat, n, lags, add_const=True, add_trend=True, add_trend2=True):
    if add_const:
        const = coefmat[:, 0]
        coefmat = coefmat[:, 1:]
    else:
        const = np.zeros([n])
    if add_trend:
        trend = coefmat[:, 0]
        coefmat = coefmat[:, 1:]
    else:
        trend = np.zeros([n])
    if add_trend2:
        trend2 = coefmat[:, 0]
        coefmat = coefmat[:, 1:]
    else:
        trend2 = np.zeros([n])

    AR = np.zeros([n, n, lags])
    for i in range(lags):
        AR[:, :, i] = coefmat[:, :n]
        coefmat = coefmat[:, n:]

    return AR, const, trend, trend2


def OLS_ReducedForm(y, lags=1, add_const=True, add_trend=False, add_trend2=False, exog=[],showresults=False):
    T, n = np.shape(y)
    if np.shape(lags) == ():
        lags = np.multiply(np.ones([n], dtype=int), int(lags), dtype=int)
    maxLags = int(np.max(lags))

    coefmat = np.empty([n, maxLags * n + (add_const+add_trend+add_trend2)])
    u = np.empty([T - maxLags, n])
    for i in range(n):
        rhs = get_rhs_flex(y, lags[i], maxLags, add_const=add_const, add_trend=add_trend, add_trend2=add_trend2)
        lhs = y[maxLags:, i]
        if np.array(exog).size != 0:
            rhs = np.append(rhs, np.transpose(np.array([exog[lags[i]: ]])), axis=1)

        try:
            model = sm.OLS(lhs, rhs)
            results = model.fit()
            if showresults:
                print(results.summary())

            if np.array(exog).size != 0:
                results.params = results.params[:-1]
            coefmat[i, :] = results.params

            u[:, i] = results.resid
        except:
            coefmat = []
            u[:,i] = y[:,i]

    AR, const, trend, trend2 = get_ARMAtrices(coefmat, n, maxLags, add_const=add_const, add_trend=add_trend, add_trend2=add_trend2)
    out = dict()
    out['u'] = u
    out['AR'] = AR
    out['const'] = const
    out['trend'] = trend
    out['trend2'] = trend2
    return out


def infocrit(y, maxLag, add_const=True, add_trend=False, add_trend2=False):
    T, n = np.shape(y)
    T = T - maxLag

    par_plus = int(add_const) + int(add_trend) + int(add_trend2)

    AIC_crit = np.full([maxLag + 1, 2], 0.)
    AIC_crit_row = np.full([n, maxLag + 1, 2], 0.)
    BIC_crit = np.full([maxLag + 1, 2], 0.)
    BIC_crit_row = np.full([n, maxLag + 1, 2], 0.)
    for this_lag in range(maxLag + 1):
        thisY = y[maxLag - this_lag:, :]

        out_redform = OLS_ReducedForm(thisY, this_lag, add_const=add_const, add_trend=add_trend,
                                      add_trend2=add_trend2)
        u = out_redform['u']
        this_aic = np.log(np.linalg.det(np.matmul(np.transpose(u), u) / T)) + 2 / T * (
                np.sum(this_lag) * np.power(n, 2) + par_plus * n)
        AIC_crit[this_lag, :] = np.array([this_lag, this_aic])

        this_BIC = np.log(np.linalg.det(np.matmul(np.transpose(u), u) / T)) + np.log(T) / T * (
                np.sum(this_lag) * np.power(n, 2) + par_plus * n)
        BIC_crit[this_lag, :] = np.array([this_lag, this_BIC])

        for i in range(n):
            this_aic_row = np.log(np.matmul(np.transpose(u[:, i]), u[:, i]) / T) + 2 / T * (
                    np.sum(this_lag) * n + par_plus)
            AIC_crit_row[i, this_lag, :] = np.array([this_lag, this_aic_row])

            this_bic_row = np.log(np.matmul(np.transpose(u[:, i]), u[:, i]) / T) + np.log(T) / T * (
                    np.sum(this_lag) * n + par_plus)
            BIC_crit_row[i, this_lag, :] = np.array([this_lag, this_bic_row])

    AIC_min_crit = np.argmin(AIC_crit[:, 1])
    BIC_min_crit = np.argmin(BIC_crit[:, 1])
    AIC_min_crit_row = np.zeros([n])
    BIC_min_crit_row = np.zeros([n])
    for i in range(n):
        AIC_min_crit_row[i] = np.argmin(AIC_crit_row[i, :, 1])
        BIC_min_crit_row[i] = np.argmin(BIC_crit_row[i, :, 1])

    out = dict()
    out['AIC_crit'] = AIC_crit
    out['AIC_crit_row'] = AIC_crit_row
    out['AIC_min_crit'] = AIC_min_crit
    out['AIC_min_crit_row'] = AIC_min_crit_row
    out['BIC_crit'] = BIC_crit
    out['BIC_crit_row'] = BIC_crit_row
    out['BIC_min_crit'] = BIC_min_crit
    out['BIC_min_crit_row'] = BIC_min_crit_row
    return out

def get_FEVD(u,B,AR,horizon):
    [T,n] = np.shape(u)

    e = SVAR.innovation(u, SVAR.get_BVector(B))
    scale = np.dot(np.transpose(e), e) / (T  - 1)
    B = np.dot(B, np.sqrt(np.diag(np.diag(scale))))

    [irf, phi] = SVAR.get_IRF(B, AR, irf_length=horizon, scale=False,outPhi=True)

    fevd = np.full([n,n],np.nan)
    for j in range(n):
        VarEj = np.sum(np.power(phi[j,:,:],2))
        for k in range(n):
            VarEjk = np.sum(np.power(phi[j,k,:],2))
            fevd[j,k] = VarEjk/VarEj

    return fevd

def get_IRF(B, AR, irf_length=12, scale=False, outPhi=False):
    n = np.shape(B)[0]

    if np.ndim(AR) == 2:
        lags = 1
        AR_new = np.full([n, n, 1], np.nan)
        AR_new[:, :, 0] = AR
        AR = AR_new
    else:
        lags = np.shape(AR)[2]
    phi = np.full([n, n, irf_length], np.nan)

    phi[:, :, 0] = np.eye(n, n)

    # normalize impact
    if scale:
        B = np.matmul(B, np.linalg.inv(np.diag(np.diag(B))))

    for i in range(1, irf_length):
        tmpsum = np.zeros([n, n])
        for j in range(i):
            if j < lags:
                tmpsum = tmpsum + np.matmul(phi[:, :, i - j - 1], AR[:, :, j])
        phi[:, :, i] = tmpsum

    for i in range(irf_length):
        phi[:, :, i] = np.matmul(phi[:, :, i], B)

    irf = np.full([irf_length, n, n], np.nan)
    for i in range(n):
        for j in range(irf_length):
            irf[j, :, i] = phi[:, i, j]
    if outPhi:
        out = irf, phi
    else:
        out = irf
    return out



def plot_IRF(irf, irf_bootstrap=[], alphas=np.array([0.2 / 2, 0.32 / 2]), shocks=[], responses=[], shocknames=[], responsnames=[], cummulative=[], sym=False):
    n = np.shape(irf)[1]
    if np.array(shocks).size == 0:
        shocks = np.arange(n)
    if np.array(responses).size == 0:
        responses = np.arange(n)
    if np.array(shocknames).size == 0:
        shocknames = np.arange(n)
    if np.array(responsnames).size == 0:
        responsnames = np.array([f"y_{{{i},t}}" for i in range(1, n + 1)])
    if np.array(cummulative).size == 0:
        cummulative = np.full(n, False)

    num_rows = np.size(shocks)
    num_cols = np.size(responses)
    fig_width = 6
    fig_height = fig_width * num_rows / num_cols


    fig, ax = plt.subplots(np.size(shocks), np.size(responses),   sharex=True,tight_layout=True, figsize=(fig_width, fig_height))


    plotcounter = 1

    for response_idx, response  in enumerate(responses):

        for shock_idx, shock in enumerate(shocks):
            y = irf[:, response, shock]
            try:
                y_bootstrap = np.asarray(irf_bootstrap)[:,:,response,shock]
                if cummulative[response]:
                    y_new = copy.deepcopy(y)
                    y_bootstrap_new = copy.deepcopy(y_bootstrap)
                    for t in range(np.size(y)):
                        y_new[t] = np.sum(y[0:t + 1])
                        y_bootstrap_new[:,t]= np.sum(y_bootstrap[:,0:t+1],axis=1)
                    y = copy.deepcopy(y_new)
                    y_bootstrap = y_bootstrap_new
            except:
                if cummulative[response]:
                    y_new = copy.deepcopy(y)
                    for t in range(np.size(y)):
                        y_new[t] = np.sum(y[0:t + 1])
                    y = copy.deepcopy(y_new)

            x = np.arange(np.shape(y)[0])


            ax[ shock_idx,response_idx].plot(x, y, '--',color="blue")
            ax[shock_idx,response_idx].axhline(0, color='black', linewidth=0.5, )
            # Plot Bootstrap irf quantiles
            for alpha in alphas:
                if np.array(irf_bootstrap).size != 0:
                    if sym:
                        alpha_n = 1 - alpha
                        s = np.sqrt(np.var(y_bootstrap, axis=0))
                        lower = y - norm.ppf(alpha_n) * s
                        upper = y + norm.ppf(alpha_n) * s

                    else:
                        lower = np.quantile(y_bootstrap, alpha, axis=0)
                        upper = np.quantile(y_bootstrap, 1 - alpha, axis=0)



                    ax[shock_idx,response_idx].fill_between(x, lower, upper, color='blue', alpha=0.25)

                ylab  = r'$\epsilon^{' + str(shocknames[ shock]) + '}$'
                xlab = r'$ ' + str(responsnames[response ]) + '$'

                if shock == shocks[-1]:
                    ax[shock_idx, response_idx  ].set(xlabel = xlab)
                if response == responses[0]:
                    ax[shock_idx, response_idx  ].set(ylabel= ylab)


            plotcounter += 1

    plt.show()

    return fig





def summarize_shocks(u,varnames):
    T,n = np.shape(u)
    TabData = list()
    TabData.append(np.append("Mean", np.round(np.mean(u, axis=0), 3)))
    TabData.append(np.append("Median", np.round(np.median(u, axis=0), 3)))
    TabData.append(np.append("Std. deviation", np.round(np.std(u, axis=0), 3)))
    TabData.append(np.append("Variance", np.round(np.var(u, axis=0), 3)))
    TabData.append(np.append("Skewness", np.round(scipy.stats.skew(u, axis=0), 3)))
    TabData.append(np.append("Kurtosis", np.round(scipy.stats.kurtosis(u, axis=0, fisher=False), 3)))
    JBTest = np.zeros(n)
    for i in range(n):
        JBTest[i] = np.round(scipy.stats.jarque_bera(u[:, i])[1], 3)
    TabData.append(np.append("Jarque-Bera Test (p-value)", JBTest))
    tableSummary = tabulate(TabData, headers=varnames, tablefmt="github")
    print(tableSummary)
    return TabData

def transform_SVARbootstrap_out(out_bootstrap):
    out_boootstrap_irf = np.full([len(out_bootstrap),
                                  np.shape(out_bootstrap[0][2])[0],
                                  np.shape(out_bootstrap[0][2])[1],
                                  np.shape(out_bootstrap[0][2])[2]], 0.0)
    out_bootstrap_B = np.full([len(out_bootstrap),
                               np.shape(out_bootstrap[0][1])[0],
                               np.shape(out_bootstrap[0][1])[1]], 0.0)
    out_bootstrap_AR = np.full([len(out_bootstrap),
                                np.shape(out_bootstrap[0][0])[0],
                                np.shape(out_bootstrap[0][0])[1],
                                np.shape(out_bootstrap[0][0])[2]], 0.0)
    for i in range(len(out_bootstrap)):
        out_boootstrap_irf[i, :, :, :] = out_bootstrap[i][2]
        out_bootstrap_B[i, :, :] = out_bootstrap[i][1]
        out_bootstrap_AR[i, :, :, :] = out_bootstrap[i][0]

    return out_boootstrap_irf, out_bootstrap_B, out_bootstrap_AR