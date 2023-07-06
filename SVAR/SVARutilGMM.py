import copy
import itertools
from types import FunctionType

import numpy as np

import SVAR
import scipy.stats

def get_Cr(r, n):
    # Generates indices of co-moment
    # Creat an array of all possible combinations of n numbers from 0 to n that sum up to r
    Cr_tmp = list(itertools.combinations_with_replacement(range(r), n))
    Cr = list()
    for cr in Cr_tmp:
        if sum(cr) == r:
            for this_cr in np.unique(list(itertools.permutations(cr)), axis=0):
                Cr.append(this_cr)
    Cr = np.asarray(Cr)

    # # improved code
    # Generates all possible combinations of n numbers from 0 to n that sum up to r
    # Cr_tmp = itertools.combinations_with_replacement(range(r), n)
    # Cr = []
    # used = set()
    # for cr in Cr_tmp:
    #     if sum(cr) == r:
    #         for this_cr in itertools.permutations(cr):
    #             if this_cr not in used:
    #                 Cr.append(this_cr)
    #                 used.add(this_cr)
    # Cr = np.asarray(Cr)
    return Cr






def get_Mr(r, n):
    # Generates indices of moments (e.g. variance)

    Mr = np.dot(np.eye(n), r)
    Mr = Mr.astype(int)
    return Mr

def get_Moments_MIcorrection(n, blocks=False):
    if blocks == False:
        blocks = list()
        blocks.append(np.array([1, n]))
    moments = np.full([1, n], 0)
    for block in blocks:
        n_min = block[0]
        n_max = block[1]
        block_length = 1 + n_max - n_min
        if block_length > 1:
            moments_this = get_Cr(4, block_length)
            select = np.full(np.shape(moments_this)[0], False)
            for idx, moment in enumerate(moments_this):
                if np.isin(2, moment) and not(np.isin(1, moment)):
                    select[idx] = True
            moments_this = moments_this[select]


            if np.shape(moments_this)[0] != 0:
                number_moments_this = np.shape(moments_this)[0]
                moments_this = np.hstack([np.zeros([number_moments_this, n_min - 1], dtype=int), moments_this])
                moments_this = np.hstack([moments_this, np.zeros([number_moments_this, n - n_max], dtype=int)])
                moments = np.append(moments, moments_this, axis=0)

    moments = moments[1:, :]
    return moments

def get_Moments(estimator, n, blocks=False, addThirdMoments=False, addFourthMoments=True,  moments_MeanIndep=False):
    if blocks == False:
        blocks = list()
        blocks.append(np.array([1, n]))
    moments = np.full([1, n], 0)

    if estimator == 'GMM' or estimator == 'CUE' or estimator == 'CSUE':
        moments = np.append(moments, get_Mr(2, n), axis=0)
        moments = np.append(moments, get_Cr(2, n), axis=0)

    for block in blocks:
        n_min = block[0]
        n_max = block[1]
        block_length = 1 + n_max - n_min
        if block_length > 1:
            moments_this = np.full([1, block_length], 0)
            if addThirdMoments:
                if estimator == 'GMM' or estimator == 'CUE' or estimator == 'CSUE':
                    moment_add = get_Cr(3, block_length)
                    moments_this = np.append(moments_this, moment_add, axis=0)

                elif estimator == 'GMM_W':
                    moment_add = get_Cr(3, block_length)

                    moments_this = np.append(moments_this, moment_add, axis=0)

                elif estimator == 'GMM_WF' or estimator == 'RidgeBWF':
                    moments_this = np.append(moments_this, get_Mr(3, block_length), axis=0)

            if addFourthMoments:
                if estimator == 'GMM' or estimator == 'CUE' or estimator == 'CSUE':
                    moment_add = get_Cr(4, block_length)

                    if moments_MeanIndep:
                        select = np.full(np.shape(moment_add)[0], False)
                        for idx, moment in enumerate(moment_add):
                            if np.isin(3, moment) or (np.isin(1, moment)):
                                select[idx] = True
                        moment_add = moment_add[select]

                    moments_this = np.append(moments_this, moment_add, axis=0)


                elif estimator == 'GMM_W':
                    moment_add = get_Cr(4, block_length)

                    if moments_MeanIndep:
                        select = np.full(np.shape(moment_add)[0], False)
                        for idx, moment in enumerate(moment_add):
                            if np.isin(3, moment) or (np.isin(1, moment)):
                                select[idx] = True
                        moment_add = moment_add[select]

                    moments_this = np.append(moments_this, moment_add, axis=0)

                elif estimator == 'GMM_WF' or estimator == 'RidgeBWF':
                    moments_this = np.append(moments_this, get_Mr(4, block_length), axis=0)
            moments_this = moments_this[1:, :]

            if np.shape(moments_this)[0] != 0:
                number_moments_this = np.shape(moments_this)[0]
                moments_this = np.hstack([np.zeros([number_moments_this, n_min - 1], dtype=int), moments_this])
                moments_this = np.hstack([moments_this, np.zeros([number_moments_this, n - n_max], dtype=int)])
                moments = np.append(moments, moments_this, axis=0)

    moments = moments[1:, :]
    return moments

def get_Moments_RecNonRed(n, addThirdMoments=False, addFourthMoments=True):
    moments = get_Moments('GMM', n, blocks=False, addThirdMoments=addThirdMoments, addFourthMoments=addFourthMoments)

    # unselect redundant moments
    select = np.full(np.shape(moments)[0], False)
    for idx, moment in enumerate(moments):
        if np.sum(moment) == 2:
                select[idx] = True
        elif np.sum(moment)==3:
            if np.isin(2,moment):
                select[idx] = True
        elif np.sum(moment)==4:
            if np.isin(3,moment):
                select[idx] = True
            elif not(np.isin(1,moment)):
                select[idx] = True

    moments = moments[select,:]



    return moments


def get_Moments_powerindex(moments):
    moments_fast = np.zeros([np.shape(moments)[0], np.shape(moments)[1] * 5], dtype=bool)
    counter = 0
    for moment in moments:
        for i in range(np.shape(moments)[1]):
            moments_fast[counter, i * 5 + moment[i]] = 1
        counter += 1
    return moments_fast



def get_Moment_transformed(moment):
    moment_trans = np.array([], dtype=int)
    for z in range(np.size(moment)):
        moment_trans = np.append(moment_trans, np.ones(moment[z], dtype=int) * (z + 1))
    return moment_trans



def calc_f(moments, moments_powerindex,e_power,T):
    # # Old code
    # counter = 0
    # f = np.empty([T, np.shape(moments)[0]])
    # for mom_fast in moments_powerindex:
    #     f[:, counter] = np.prod(e_power[:, mom_fast], axis=1)
    #     if ~np.isin(1, moments[counter, :]):
    #         if np.isin(2, moments[counter, :]):
    #             if np.sum(moments[counter, :]) == 2:
    #                 f[:, counter] = np.subtract(f[:, counter], 1)
    #             elif np.sum(moments[counter, :]) == 4:
    #                 f[:, counter] = np.subtract(f[:, counter], np.power(1,2))
    #         if np.isin(4, moments[counter, :]):
    #             f[:, counter] = np.subtract(f[:, counter], 3)
    #     counter += 1

    f = calc_f_improved(moments, moments_powerindex,e_power,T)
    return f

def calc_f_improved(moments, moments_powerindex,e_power,T):

    f = np.empty([T, np.shape(moments)[0]])
    # Iterate over the fast moments
    for i, mom_fast in enumerate(moments_powerindex):
        # Compute the powers of the exponents for the current moment
        powers = e_power[:, mom_fast]
        # Compute the products of the powers along axis 1
        f[:, i] = np.prod(powers, axis=1)


    k_array = np.zeros(moments.shape[0])
    k_array[~np.any(moments == 1, axis=1) & (np.sum(moments, axis=1)==2)] = 1
    k_array[~np.any(moments == 1, axis=1) & (np.sum(moments, axis=1)==4) & np.any(moments == 2, axis=1) ] = 1
    k_array[~np.any(moments == 1, axis=1) & np.any(moments == 4, axis=1)  ] = 3
    f = f - k_array
    return f



def get_f(u, b, restrictions, moments, moments_powerindex, whiten=False, blocks=False  ):
    T, n = np.shape(u)
    e = SVAR.innovation(u, b, restrictions=restrictions, whiten=whiten, blocks=blocks )
    e_power = np.empty([np.shape(e)[0], n * 5])
    e_power[:, range(0, 5 * n, 5)] = np.ones([np.shape(e)[0], n])
    for i in range(n):
        for j in range(1, 5):
            this_entry = (i) * 5 + j
            e_power[:, this_entry] = np.multiply(e_power[:, this_entry - 1], e[:, i])

    f = calc_f(moments, moments_powerindex,e_power,T)


    return f

def get_f_wf(u, b, restrictions, moments, moments_powerindex, whiten=True, blocks=False ):
    T, n = np.shape(u)
    e = SVAR.innovation(u, b, restrictions=restrictions, whiten=whiten, blocks=blocks )


    f = np.empty([T, np.shape(moments)[0]])
    # Iterate over the fast moments
    for i, mom in enumerate(moments):
        # Compute the powers of the exponents for the current moment
        powers = np.power(e,mom)
        # Compute the products of the powers along axis 1
        f[:, i] = np.prod(powers, axis=1)

    k_array = np.zeros(moments.shape[0])
    k_array[~np.any(moments == 1, axis=1) & (np.sum(moments, axis=1)==2)] = 1
    k_array[~np.any(moments == 1, axis=1) & (np.sum(moments, axis=1)==4) & np.any(moments == 2, axis=1) ] = 1
    k_array[~np.any(moments == 1, axis=1) & np.any(moments == 4, axis=1)  ] = 3
    f = f - k_array

    return f


def get_g(u, b, restrictions, moments, moments_powerindex, whiten=False, blocks=False  ):
    return np.mean(
        get_f(b=b[:], u=u[:], restrictions=restrictions[:], moments=moments[:], moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks  ), axis=0)

def get_g_wf(u, b, restrictions, moments, moments_powerindex, whiten=True, blocks=False  ):
    return np.mean(
        get_f_wf(b=b[:], u=u[:], restrictions=restrictions[:], moments=moments[:], moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks ), axis=0)

def generat_S_del_ql_function(moments,q,l):
    n = np.shape(moments)[1]

    function_string = str("def f(omegaext,A): ")
    function_string = function_string + str('S=np.array([')
    for countx, moment_x in enumerate(moments):
        if countx != 0:
            function_string = function_string + str(',')

        this_moment_string = str('[ ')
        for county, moment_y in enumerate(moments):
            if county != 0:
                this_moment_string = this_moment_string + str(',')

            moment_this = moment_x + moment_y

            omega_str = SVAR.SVARutilGMM.get_omegaextstring(moment_this, n, q, l)

            if ~np.isin(1, moment_x):
                omega_str2 = SVAR.SVARutilGMM.get_omegaextstring(moment_y, n, q, l)
                omega_str = omega_str + '-' + omega_str2
            if ~np.isin(1, moment_y):
                omega_str2 = SVAR.SVARutilGMM.get_omegaextstring(moment_x, n, q, l)
                omega_str = omega_str + '-' + omega_str2
            if ~np.isin(1, moment_y) and ~np.isin(1, moment_x):
                omega_str = omega_str + '+ 1'

            this_moment_string = this_moment_string + omega_str
        this_moment_string = this_moment_string + str(']')
        function_string = function_string + this_moment_string
    function_string = function_string + str(']);')
    function_string = function_string + str('return S')

    function_code = compile(function_string, "<string>", "exec")
    function_func = FunctionType(function_code.co_consts[0], globals(), "f")

    return function_func


def generat_S_del_functions(moments,restrictions):
    n = np.shape(moments)[1]
    Sdelbql_func_all = list()
    for q in range(n):
        for l in range(n):
            # only if B element is not restricted
            if np.isnan(restrictions[q, l]):
                Sdelbql_func_all.append(SVAR.SVARutilGMM.generat_S_del_ql_function(moments, q, l))

    return Sdelbql_func_all

def generate_Jacobian_function(moments, restrictions):
    n = np.shape(moments)[1]
    function_string = str("def f(u,b,restrictions,CoOmega): ")
    function_string = function_string + str('B = SVAR.get_BMatrix(b, restrictions=restrictions);')
    function_string = function_string + str('A = np.linalg.inv(B);')

    function_string = function_string + str('[CoOmega1, CoOmega2, CoOmega3, CoOmega4, CoOmega5, CoOmega6, CoOmega7, CoOmega8] = CoOmega;')

    function_string = function_string + str('G=np.array([')
    counter = 0
    for moment in moments:
        counter_inner = 0
        if counter != 0:
            function_string = function_string + str(',')
        # Calculate Jacboian of this moment
        this_moment_string = str('[ ')
        for q in range(0, n):
            for l in range(0, n):
                # only if B element is not restricted
                if np.isnan(restrictions[q, l]):
                    tmp_string = momentderiv(moment,q,l,n)

                    if counter_inner != 0:
                        this_moment_string = this_moment_string + str(',')
                    counter_inner += 1
                    this_moment_string = this_moment_string + tmp_string
        this_moment_string = this_moment_string + str(']')

        # Append to function_string
        function_string = function_string + this_moment_string
        counter += 1

    function_string = function_string + str(']);')
    function_string = function_string + str('return G')
    # print(function_string)

    function_code = compile(function_string, "<string>", "exec")
    function_func = FunctionType(function_code.co_consts[0], globals(), "f")

    return function_func

def generate_SIndep_function(moments):
    function_string = str("def f(omega): ")
    function_string = function_string + str('S=np.array([')
    for countx, moment_x in enumerate(moments):
        if countx != 0:
            function_string = function_string + str(',')

        this_moment_string = str('[ ')
        for county, moment_y in enumerate(moments):
            if county != 0:
                this_moment_string = this_moment_string + str(',')

            # symmetric matrix
            if countx>county:
                omega_str = '0'
            else:
                moment_this = moment_x + moment_y

                omega_str = SVAR.SVARutilGMM.get_omegastring(moment_this)

                if ~np.isin(1, moment_x):
                    omega_str2 = SVAR.SVARutilGMM.get_omegastring(moment_y)
                    omega_str = omega_str + '-' + omega_str2
                if ~np.isin(1, moment_y):
                    omega_str2 = SVAR.SVARutilGMM.get_omegastring(moment_x)
                    omega_str = omega_str + '-' + omega_str2
                if ~np.isin(1, moment_y) and ~np.isin(1, moment_x):
                    omega_str = omega_str + '+ 1'

            this_moment_string = this_moment_string + omega_str
        this_moment_string = this_moment_string + str(']')
        function_string = function_string + this_moment_string
    function_string = function_string + str(']);')
    function_string = function_string + str('return S')
    # print(function_string)
    function_code = compile(function_string, "<string>", "exec")
    function_func = FunctionType(function_code.co_consts[0], globals(), "f")

    return function_func


def eval_CoOmega(moment,CoOmega):
    [CoOmega1, CoOmega2, CoOmega3, CoOmega4, CoOmega5, CoOmega6, CoOmega7, CoOmega8] = CoOmega

    moment_trans = get_Moment_transformed(moment)
    if sum(moment) == 0:
        omega = 1
    elif sum(moment) == 1:
        omega = CoOmega1[moment_trans[0]  - 1     ]
    elif sum(moment) == 2:
        omega = CoOmega2[moment_trans[0]  - 1  , moment_trans[1]  - 1  ]
    elif sum(moment) == 3:
        omega = CoOmega3[moment_trans[0]  - 1  , moment_trans[1]  - 1, moment_trans[2]  - 1   ]
    elif sum(moment) == 4:
        omega = CoOmega4[moment_trans[0]  - 1  , moment_trans[1]  - 1, moment_trans[2]  - 1, moment_trans[3]  - 1    ]
    elif sum(moment) == 5:
        omega = CoOmega5[moment_trans[0] - 1, moment_trans[1] - 1, moment_trans[2] - 1, moment_trans[3] - 1, moment_trans[4] - 1]
    elif sum(moment) == 6:
        omega = CoOmega6[moment_trans[0] - 1, moment_trans[1] - 1, moment_trans[2] - 1, moment_trans[3] - 1, moment_trans[4] - 1, moment_trans[5] - 1]
    elif sum(moment) == 7:
        omega = CoOmega7[moment_trans[0] - 1, moment_trans[1] - 1, moment_trans[2] - 1, moment_trans[3] - 1, moment_trans[4] - 1, moment_trans[5] - 1, moment_trans[6] - 1]
    elif sum(moment) == 8:
        omega = CoOmega8[moment_trans[0] - 1, moment_trans[1] - 1, moment_trans[2] - 1, moment_trans[3] - 1, moment_trans[4] - 1, moment_trans[5] - 1, moment_trans[6] - 1, moment_trans[7] - 1]

    else:
        print("error")

    return omega

def get_CoOmegaEvalString(moment):
    moment_trans = get_Moment_transformed(moment)
    if sum(moment) == 0:
        omega_str = '1'
    elif sum(moment) == 1:
        omega_str = 'CoOmega1[' + str(int(moment_trans[0]) - 1) +   ']'
    elif sum(moment) == 2:
        omega_str = 'CoOmega2[' + str(int(moment_trans[0]) - 1) + ', ' + str(
            int(moment_trans[1]) - 1) + ']'
    elif sum(moment) == 3:
        omega_str = 'CoOmega3[' + str(int(moment_trans[0]) - 1) + ', ' + str(
            int(moment_trans[1]) - 1) + ', ' + str(int(moment_trans[2]) - 1) + ']'
    elif sum(moment) == 4:
        omega_str = 'CoOmega4[' + str(int(moment_trans[0]) - 1) + ', ' + str(
            int(moment_trans[1]) - 1) + ', ' + str(
            int(moment_trans[2]) - 1) + ', ' + str(int(moment_trans[3]) - 1) + ']'
    elif sum(moment) == 5:
        omega_str = 'CoOmega5[' \
                    + str(int(moment_trans[0]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[1]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[2]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[3]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[4]) - 1) + ']'
    elif sum(moment) == 6:
        omega_str = 'CoOmega6[' \
                    + str(int(moment_trans[0]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[1]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[2]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[3]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[4]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[5]) - 1)  + ']'
    elif sum(moment) == 7:
        omega_str = 'CoOmega7[' \
                    + str(int(moment_trans[0]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[1]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[2]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[3]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[4]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[5]) - 1)\
                    + ', ' \
                    + str(int(moment_trans[6]) - 1)   + ']'
    elif sum(moment) == 8:
        omega_str = 'CoOmega8[' \
                    + str(int(moment_trans[0]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[1]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[2]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[3]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[4]) - 1) \
                    + ', ' \
                    + str(int(moment_trans[5]) - 1)\
                    + ', ' \
                    + str(int(moment_trans[6]) - 1)\
                    + ', ' \
                    + str(int(moment_trans[7]) - 1)    + ']'
    else:
        omega_str = '999'
    return omega_str


def get_omegastring(moment):
    omega_str = ''
    if sum(moment) == 0:
        omega_str = '1'
    else:
        for countmom, mom in enumerate(moment):
            if mom > 0:
                if not (omega_str == ''):
                    omega_str = omega_str + '*'
                omega_str = omega_str + 'omega[' + str(int(countmom)) + ', ' + str(int(mom) - 1) + ']'

    return omega_str

def get_omegaextstring(moment,n,q,l):
    omega_str = ''
    for k, mom in enumerate(moment):
        if k > 0:
            omega_str = omega_str + '+'
        for idx in range(n):
            if idx > 0:
                omega_str = omega_str + '*'

            if idx == k:
                if moment[idx] == 0:
                    omega_str = omega_str + str(0)
                else:
                    omega_str = omega_str + str(int(mom)) +'*'+ 'A[' + str(k) + ',' + str(q) +']*' + 'omegaext[' + str(int(idx)) + ', ' + str(int(moment[idx]) - 1) + ', ' + str(l+1) + ']'
            else:
                omega_str = omega_str + 'omegaext[' + str(int(idx)) + ', ' + str(int(moment[idx])  ) + ', 0'  + ']'


    return omega_str


def momentderiv(moment,q,l,n):
    tmp_string = str(' ')

    for k in range(0, n):
        if moment[k] != 0:
            moment_new = copy.deepcopy(moment)
            moment_new[l] = moment_new[l] + 1
            moment_new[k] = moment_new[k] - 1

            omega_str = get_CoOmegaEvalString(moment_new)

            if moment[k] != 0:
                tmp_string = tmp_string + str(
                    '-A[' + str(k) + ',' + str(q) + '] * ' + str(moment[k]) + ' * ' + omega_str)
    return tmp_string

def get_G_Indep(Moments, B, omega, B_restrictions=[]):
    # Calculates analytical G = E[ partial f(u,B_0) / partial B ]

    n, n = np.shape(B)

    if np.array(B_restrictions).size == 0: B_restrictions = np.full([n, n], np.nan)

    A = np.linalg.inv(B)
    n = np.shape(B)[1]
    # add one row to omega for moment e^0=1
    omega = np.concatenate((np.ones([n, 1]), omega), axis=1)
    # empty G array
    G = np.empty([np.shape(Moments)[0], np.sum(np.isnan(B_restrictions))])
    moment_counter = 0
    for moment in Moments:
        elementcounter = 0
        for i in range(0, n):
            for j in range(0, n):
                # only if B element is not restricted
                if np.isnan(B_restrictions[i, j]):
                    G_this = 0
                    for idx in range(0, n):
                        moment_new = copy.deepcopy(moment)
                        moment_new[j] = moment_new[j] + 1
                        moment_new[idx] = moment_new[idx] - 1
                        G_this = G_this - A[idx, i] * moment[idx] * np.prod(omega[np.array([range(0, n)]), moment_new])
                    G[moment_counter, elementcounter] = G_this
                    elementcounter = elementcounter + 1
        moment_counter = moment_counter + 1
    return G



def get_G(Moments, B,  CoOmega, B_restrictions=[]):
    # Calculates analytical G = E[ partial f(u,B_0) / partial B ]

    n, n = np.shape(B)

    if np.array(B_restrictions).size == 0: B_restrictions = np.full([n, n], np.nan)

    A = np.linalg.inv(B)
    n = np.shape(B)[1]
    # empty G array
    G = np.empty([np.shape(Moments)[0], np.sum(np.isnan(B_restrictions))])
    moment_counter = 0
    for moment in Moments:
        elementcounter = 0
        for i in range(0, n):
            for j in range(0, n):
                # only if B element is not restricted
                if np.isnan(B_restrictions[i, j]):
                    G_this = 0
                    for idx in range(0, n):
                        moment_new = copy.deepcopy(moment)
                        moment_new[j] = moment_new[j] + 1
                        moment_new[idx] = moment_new[idx] - 1

                        # catch negative values. Unimportant derivative is zero
                        if moment_new[idx] < 0:
                            moment_new[idx] = 0

                        G_this = G_this - A[idx, i] * moment[idx] * eval_CoOmega(moment_new, CoOmega)

                    G[moment_counter, elementcounter] = G_this
                    elementcounter = elementcounter + 1
        moment_counter = moment_counter + 1
    return G

def get_S_Indep(Moments_1, Moments_2, omega):
    # Calculates analytically S = E[ f_[Moments_1](u,B_0) f_[Moments_2](u,B_0)' ]

    # empty S array
    S = np.empty([np.shape(Moments_1)[0], np.shape(Moments_2)[0]])
    n = np.shape(Moments_1)[1]
    # add one row to omega for moment e^0=1
    omega = np.concatenate((np.ones([n, 1]), omega), axis=1)
    moments_1_counter = 0
    moments_2_counter = 0
    for moment_1 in Moments_1:
        for moment_2 in Moments_2:
            v_1 = - int(~np.isin(1, moment_1))
            v_2 = - int(~np.isin(1, moment_2))
            S[moments_1_counter, moments_2_counter] = np.prod(
                omega[np.array([range(0, n)]), moment_1 + moment_2]) + v_1 * np.prod(
                omega[np.array([range(0, n)]), moment_2]) + v_2 * np.prod(
                omega[np.array([range(0, n)]), moment_1]) + v_1 * v_2
            moments_2_counter = moments_2_counter + 1
        moments_2_counter = 0
        moments_1_counter = moments_1_counter + 1
    return S


def get_S(u, b,   moments, restrictions ):
    moments_powerindex = get_Moments_powerindex(moments)
    f = get_f(u=u, b=b, restrictions=restrictions, moments=moments,
              moments_powerindex=moments_powerindex)
    S = np.cov(f.T )
    return S

def get_S_uncentered(u, b,   moments, restrictions ):
    [T,n] = np.shape(u)
    moments_powerindex = get_Moments_powerindex(moments)
    f = get_f(u=u, b=b, restrictions=restrictions, moments=moments,
              moments_powerindex=moments_powerindex)
    S =  np.dot(f.T, f) / (f.shape[0] - 1)
    return S


def wald(R,r,b,avar,T):
    wald2 = avar

    wald1 = np.dot(R, b) - r
    wald3 = np.linalg.multi_dot([R, wald2, np.transpose(R)])
    try:
        wald3 = np.linalg.inv(wald3)
        wald = np.multiply(T, np.linalg.multi_dot([np.transpose(wald1), wald3, wald1]))

    except:
        wald3 = np.divide(1, wald3)
        wald = T * wald1 * wald3 * wald1

    p = 1 - scipy.stats.chi2.cdf(wald, np.shape(R)[0])

    return wald,p

def wald_param_all(B,restrictions,avar,T):
    b = SVAR.get_BVector(B, restrictions=restrictions, whiten=False)
    wald_all = np.empty_like(b)
    wald_all_p = np.empty_like(b)
    avar = avar[np.logical_not(np.isnan(avar))]
    avar = np.reshape(avar, [np.size(b), np.size(b)])
    for idx, b_this in np.ndenumerate(b):
        R = np.full([1, np.size(b)], 0)
        R[0, idx] = 1
        r = 0
        wald_all[idx], wald_all_p[idx] = SVAR.SVARutilGMM.wald(R, r, b, avar, T)
    return wald_all, wald_all_p

def waldRec(B,avar,restrictions,T):
    restrictions_rec = SVAR.SVARutil.getRestrictions_recursive(B)
    b =  SVAR.get_BVector(B, restrictions=restrictions, whiten=False)
    select = (np.isnan(SVAR.get_BVector(restrictions_rec))) == False
    select = select[(np.isnan(SVAR.get_BVector(restrictions)))]
    counter = 0
    R = np.full([np.sum(select), np.size(b)], 0)
    for idx, b_this in np.ndenumerate(b):
        if select[idx]:
            R[counter, idx] = 1
            counter += 1
    r = np.full([np.sum(select)], 0)
    avar = avar[np.logical_not(np.isnan(avar))]
    avar = np.reshape(avar, [np.size(b), np.size(b)])
    wald_rec, wald_rec_p = SVAR.SVARutilGMM.wald(R, r, b, avar, T)
    return wald_rec, wald_rec_p

def waldRest(B,avar,restrictions,T,testrest):
    b = SVAR.get_BVector(B, restrictions=restrictions, whiten=False)
    select = (np.isnan(SVAR.get_BVector(testrest))) == False
    select = select[(np.isnan(SVAR.get_BVector(restrictions)))]
    counter = 0
    R = np.full([np.sum(select), np.size(b)], 0)
    for idx, b_this in np.ndenumerate(b):
        if select[idx]:
            R[counter, idx] = 1
            counter += 1
    r = np.full([np.sum(select)], SVAR.get_BVector(testrest[np.isnan(restrictions)])[select])
    avar = avar[np.logical_not(np.isnan(avar))]
    avar = np.reshape(avar, [np.size(b), np.size(b)])
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.wald(R, r, b, avar, T)
    return wald_stat, wald_stat_p

def LRRest(u,SVAR_outsave,testrest):
    options = copy.deepcopy(SVAR_outsave['options'])
    restrictions = options['restrictions']
    restrictions[np.isnan(restrictions)] = testrest[np.isnan(restrictions)]
    options['restrictions'] = restrictions
    options['bstart'] = options['bstart'][np.isnan(SVAR.get_BVector(restrictions))]
    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=options['moments'], restrictions=restrictions)
    options['Jacobian'] = Jacobian

    SVAR_outrest = SVAR.SVARest(u, estimator=SVAR_outsave['options']['estimator'], options=options,prepared=True)
    lr_stat = options['T'] * (SVAR_outrest['loss'] - SVAR_outsave['loss'])
    lr_stat_p = 1 - scipy.stats.chi2.cdf(lr_stat, np.sum((np.isnan(SVAR.get_BVector(testrest))) == False))
    return lr_stat, lr_stat_p


def LMRest(u, SVAR_outsave, testrest):
    options = copy.deepcopy(SVAR_outsave['options'])
    restrictions = options['restrictions']
    restrictions[np.isnan(restrictions)] = testrest[np.isnan(restrictions)]
    options['restrictions'] = restrictions
    options['bstart'] = options['bstart'][np.isnan(SVAR.get_BVector(restrictions))]
    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=options['moments'], restrictions=restrictions)
    options['Jacobian'] = Jacobian

    SVAR_outrest = SVAR.SVARest(u, estimator=SVAR_outsave['options']['estimator'], options=options, prepared=True)

    options_unrest = SVAR_outsave['options']
    options = SVAR_outrest['options']
    CoOmega = SVAR.SVARutil.get_CoOmega(SVAR_outrest['e'])

    g = copy.deepcopy(SVAR_outsave['g']) # ToDo: Wrong? g at Brest?
    G = copy.deepcopy(SVAR_outsave['G'])
    S = copy.deepcopy(SVAR_outsave['S'])

    lm_stat1 = np.matmul(np.matmul(np.transpose(g), np.linalg.inv(S)), G)
    lm_stat2 = np.linalg.inv(np.matmul(np.matmul(np.transpose(G), np.linalg.inv(S)), G))
    lm_stat3 = np.matmul(np.matmul(np.transpose(G), np.linalg.inv(S)), g)
    lm_stat = options['T'] * np.matmul(np.matmul(lm_stat1, lm_stat2), lm_stat3)
    lm_stat_p = 1 - scipy.stats.chi2.cdf(lm_stat, np.sum((np.isnan(SVAR.get_BVector(testrest))) == False))
    return lm_stat, lm_stat_p

def get_W_opt(u, b, restrictions, moments,     Wpara='Uncorrelated',  S_func=False  ):
    if Wpara=='Uncorrelated':
        S = SVAR.SVARutilGMM.get_S(u, b=b, moments=moments,restrictions=restrictions)
    elif Wpara=='Uncorrelated_uncentered':
        S = SVAR.SVARutilGMM.get_S_uncentered(u, b=b, moments=moments,restrictions=restrictions)
    elif Wpara == 'Independent':
        e = SVAR.innovation(u,b,restrictions=restrictions)
        omega = SVAR.SVARutil.get_Omega_Moments(e)
        if S_func==False:
            S = SVAR.SVARutilGMM.get_S_Indep(Moments_1=moments, Moments_2=moments, omega=omega )
        else:
            S = S_func(omega)
            S = S + S.T - np.diag(np.diag(S))

    try:
        W = np.linalg.inv(S)
    except:
        W = np.linalg.pinv(S)

    return W
