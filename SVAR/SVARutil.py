import copy
import itertools
import numpy as np
import scipy
import SVAR.SVARutilGMM
from scipy.optimize import LinearConstraint


def innovation(u, b, restrictions=[], whiten=False, blocks=False ):
    # Calculates unmixed innovations according to: e = B^{-1} u

    B = get_BMatrix(b, restrictions=restrictions, whiten=whiten, blocks=blocks )
    A = np.linalg.inv(B)
    e = np.matmul(A, np.transpose(u))
    e = np.transpose(e)
    return e

def get_BMatrix(b, restrictions=[], whiten=False, blocks=False):
    # Transforms vectorized B into B
    if whiten:
        # ToDo: Add white restrictions
        B = get_Orthogonal(b, blocks)
    else:
        if blocks:
            restrictions = getRestrictions_blocks(blocks)

        if np.array(restrictions).size == 0:
            b_length = np.sqrt(np.size(b))
            n = b_length.astype(int)
            restrictions = np.full([n, n], np.nan)

        B = copy.deepcopy(restrictions)
        B[np.isnan(B)] = b

    return B

def get_BVector(B, restrictions=[], whiten=False, blocks=False):
    # inverse of get_BMatrix
    # ToDo: Error if element of B!=restrictions
    if whiten:
        # ToDo: Add restrictions
        b = get_Skewsym(B, blocks=blocks)
    else:
        if np.array(restrictions).size == 0:
            restrictions = np.full(np.shape(B), np.nan)
        b = B[np.isnan(restrictions) == 1]
    return b

def get_block_rec(n_rec, n):
    if not (n_rec):
        n_rec = 0
    blocks = list()
    for i in range(1, n_rec + 1):
        blocks.append(np.array([i, i]))
    if n_rec + 1 <= n:
        blocks.append(np.array([n_rec + 1, n]))
    return blocks

def getRestrictions_recursive(B):
    # restricts upper right triangular matrix
    myrestrictions = np.full(np.shape(B), np.nan)
    iu1 = np.triu_indices(np.shape(myrestrictions)[1], 1)
    myrestrictions[iu1] = B[iu1]
    return myrestrictions


def getRestrictions_blocks(blocks):
    myrestrictions = np.full([blocks[-1][-1], blocks[-1][-1]], np.nan)
    for block in blocks:
        myrestrictions[block[0] - 1:block[1], block[1]:] = 0
    return myrestrictions

def getRestrictions_none(n):
    myrestrictions = np.full([n,n], np.nan)
    return myrestrictions

def do_whitening(u, white):
    if white:
        T, n = np.shape(u)
        Sigma = np.dot(np.transpose(u), u) / T
        V = np.linalg.cholesky(Sigma)
    else:
        V = np.eye(np.size(u))
    Vinv = np.linalg.inv(V)
    z = np.matmul(Vinv, np.transpose(u))
    z = np.transpose(z)
    return z, V


def get_Skewsym(B, blocks=False):
    if not blocks:
        n = int(np.shape(B)[0])
        blocks = list()
        blocks.append(np.array([1, n]))
    s = np.array([])
    for block in blocks:
        n = block[1] - block[0] + 1
        B_this = B[block[0] - 1:block[1], block[0] - 1:block[1]]
        S = scipy.linalg.logm(B_this)
        il1 = np.tril_indices(n, k=-1)
        s = np.append(s,S[il1])

    return s


def get_Orthogonal(b, blocks=False):
    if not blocks:
        n = int(np.ceil(np.sqrt(2 * np.size(b))))
        blocks = list()
        blocks.append(np.array([1, n]))
    B = np.eye(blocks[-1][-1])

    b_counter = 0
    for block in blocks:
        n = block[1] - block[0] + 1
        S = np.full([n, n], 0.0)
        il1 = np.tril_indices(n, k=-1)
        n_s = int(np.size(il1) / 2)
        S[il1] = b[b_counter:b_counter + n_s]
        S = S - np.transpose(S)
        # B_this = scipy.linalg.expm(S)
        B_this, expm_frechet_AE = scipy.linalg.expm_frechet(S, S)
        B[block[0] - 1:block[1], block[0] - 1:block[1]] = B_this
        b_counter = b_counter + n_s
    if not (b_counter == np.shape(b)[0]):
        raise ValueError('Specified b value does not match B restrictions.')
    return B


## Permutation
def PermToB0(B0,B,avar,restrictions,T):
    b = get_BVector(B, restrictions=restrictions, whiten=False)
    n = np.shape(B0)[0]
    R = np.eye(np.shape(b)[0])
    avar = avar[np.logical_not(np.isnan(avar))]
    avar = np.reshape(avar, [np.size(b), np.size(b)])
    perms = get_AllSignPerm(n)
    score = np.zeros(np.shape(perms)[0])
    for i, perm in enumerate(perms):
        B0perm = np.matmul(B0, perm)
        if (restrictions[np.isnan(restrictions) == False] == B0perm[np.isnan(restrictions) == False]).all():
            r =get_BVector(B0perm, restrictions=restrictions, whiten=False)
            waldstat, wald_p = SVAR.SVARutilGMM.wald(R, r, b, avar, T)
            score[i] = wald_p
    Bbest = np.matmul(B, np.linalg.inv(perms[np.argmax(score)]))
    return Bbest, perms[np.argmax(score)]



def get_AllSignPerm(n):
    Perms = []

    permutations = list(itertools.permutations(range(n)))
    num_permutations = np.shape(permutations)[0]

    signs = list(itertools.product(np.array([1, -1]), repeat=n))
    num_signs = np.shape(signs)[0]

    I = np.eye(n)
    for p in range(num_permutations):
        perm_this = I[permutations[p], :]
        for s in range(num_signs):
            sign_this = np.matmul(I, np.diag(signs[s]))
            sign_perm_this = np.matmul(perm_this, sign_this)
            Perms.append(sign_perm_this)

    return Perms


def get_Omega(e):
    n = np.shape(e)[1]
    CoOmega2 = np.full([n, n], np.nan)
    CoOmega3 = np.full([n, n, n], np.nan)
    CoOmega4 = np.full([n, n, n, n], np.nan)
    for pi in range(n):
        for pj in range(pi, n):
            save_ij = np.array([np.prod(e[:, np.array([pi, pj])], axis=1)]).T
            CoOmega2[pi, pj] = np.mean(save_ij)
            for pk in range(pj, n):
                save_ijk = np.array([np.prod(np.append(save_ij, e[:, np.array([pk])], axis=1), axis=1)]).T
                CoOmega3[pi, pj, pk] = np.mean(save_ijk)
                for pl in range(pk, n):
                    save_ijkl = np.prod(np.append(save_ijk, e[:, np.array([pl])], axis=1), axis=1)
                    CoOmega4[pi, pj, pk, pl] = np.mean(save_ijkl)
    return CoOmega2, CoOmega3, CoOmega4

def get_CoOmega(e ):
    [T,n] = np.shape(e)
    CoOmega1 = np.mean(e.T,axis=1)
    CoOmega2 = np.matmul(e.T , e   )/T
    CoOmega3 = np.full([n, n, n], np.nan)
    CoOmega4 = np.full([n, n, n, n], np.nan)
    CoOmega5 = np.full([n, n, n, n,n], np.nan)
    CoOmega6 = np.full([n, n, n, n,n,n], np.nan)
    CoOmega7 = np.full([n, n, n, n,n,n,n], np.nan)
    CoOmega8 = np.full([n, n, n, n,n,n,n,n], np.nan)
    for p3 in range(n):
        e_save3 = e.T * e[:,p3]
        CoOmega3[:,:,p3] = np.matmul( e_save3    , e    )/T
        for p4 in range(n):
            e_save4 = e_save3  * e[:,p4]
            CoOmega4[:,:,p3,p4] = np.matmul(e_save4 , e  )/T
            for p5 in range(n):
                e_save5 = e_save4 * e[:,p5]
                CoOmega5[:,:,p3,p4,p5] = np.matmul(e_save5 , e  )/T
                for p6 in range(n):
                    e_save6 = e_save5 * e[:,p6]
                    CoOmega6[:, :, p3, p4, p5, p6] = np.matmul(e_save6, e ) / T
                    for p7 in range(n):
                        e_save7 = e_save6 * e[:,p7]
                        CoOmega7[:, :, p3, p4, p5, p6, p7] = np.matmul(e_save7, e ) / T
                        for p8 in range(n):
                            e_save8 = e_save7 * e[:,p8]
                            CoOmega8[:, :, p3, p4, p5, p6, p7, p8] = np.matmul(e_save8, e ) / T

    return CoOmega1, CoOmega2, CoOmega3, CoOmega4, CoOmega5, CoOmega6, CoOmega7, CoOmega8

def get_Omega_Moments(e):
    n = np.shape(e)[1]
    Omega = np.full([n, 6], np.nan)
    Omega[:, 0] = np.mean(np.power(e, 1), axis=0)
    Omega[:, 1] = np.mean(np.power(e, 2), axis=0)
    Omega[:, 2] = np.mean(np.power(e, 3), axis=0)
    Omega[:, 3] = np.mean(np.power(e, 4), axis=0)
    Omega[:, 4] = np.mean(np.power(e, 5), axis=0)
    Omega[:, 5] = np.mean(np.power(e, 6), axis=0)

    return Omega


def get_omegaext(e):
    n = np.shape(e)[1]
    omegaext = np.full([n, 7,n+1], np.nan)

    omegaext[:, 0,0] = np.mean(np.power(e, 0), axis=0)
    omegaext[:, 1,0] = np.mean(np.power(e, 1), axis=0)
    omegaext[:, 2,0] = np.mean(np.power(e, 2), axis=0)
    omegaext[:, 3,0] = np.mean(np.power(e, 3), axis=0)
    omegaext[:, 4,0] = np.mean(np.power(e, 4), axis=0)
    omegaext[:, 5,0] = np.mean(np.power(e, 5), axis=0)
    omegaext[:, 6,0] = np.mean(np.power(e, 6), axis=0)

    for i in range(n):
        omegaext[:, 0, i+1] = np.mean(np.multiply(np.power(e, 0), e[:,i].reshape(-1, 1)), axis=0)
        omegaext[:, 1, i+1] = np.mean(np.multiply(np.power(e, 1), e[:,i].reshape(-1, 1) ), axis=0)
        omegaext[:, 2, i+1] = np.mean(np.multiply(np.power(e, 2), e[:,i].reshape(-1, 1) ), axis=0)
        omegaext[:, 3, i+1] = np.mean(np.multiply(np.power(e, 3), e[:,i].reshape(-1, 1) ), axis=0)
        omegaext[:, 4, i+1] = np.mean(np.multiply(np.power(e, 4), e[:,i].reshape(-1, 1) ), axis=0)
        omegaext[:, 5, i+1] = np.mean(np.multiply(np.power(e, 5), e[:,i].reshape(-1, 1) ), axis=0)
        omegaext[:, 6, i+1] = np.mean(np.multiply(np.power(e, 6), e[:,i].reshape(-1, 1) ), axis=0)

    return omegaext
 