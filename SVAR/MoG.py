import numpy as np


def MoG_rnd(Omega, lamb, T):
    # MoG random variable

    select = np.random.uniform(0, 1, T)
    mu1 = Omega[0, 0]
    sigma1 = np.sqrt(Omega[0, 1])
    x1 = np.random.normal(loc=mu1, scale=sigma1, size=T)
    mu2 = Omega[1, 0]
    sigma2 = np.sqrt(Omega[1, 1])
    x2 = np.random.normal(loc=mu2, scale=sigma2, size=T)

    x = x1
    x[select >= lamb] = x2[select >= lamb]
    return x


def MoG_Moments(Omega, lamb):
    # Calculates first six moments of MoG

    mu1, w11 = Omega[0, :]
    mu2, w22 = Omega[1, :]

    sigma1 = np.sqrt(w11)
    sigma2 = np.sqrt(w22)

    m0 = 1

    # m1_o = lamb * mu1 + (1 - lamb) * mu2
    # m2_o = (lamb) * (np.power(mu1, 2) + w11) \
    #      + (1 - lamb) * (np.power(mu2, 2) + w22)
    # m3_o = (lamb) * (np.power(mu1, 3) + 3 * mu1 * w11) \
    #      + (1 - lamb) * (np.power(mu2, 3) + 3 * mu2 * w22)
    # m4_o = (lamb) * (np.power(mu1, 4) + 6 * np.power(mu1, 2) * w11 + 3 * np.power(w11, 2)) \
    #      + (1 - lamb) * (np.power(mu2, 4) + 6 * np.power(mu2, 2) * w22 + 3 * np.power(w22, 2))
    # m5_o = (lamb) * (np.power(mu1, 5) + 10 * np.power(mu1, 3) * w11 + 15 * mu1 * np.power(w11, 2)) \
    #      + (1 - lamb) * (np.power(mu2, 5) + 10 * np.power(mu2, 3) * w22 + 15 * mu2 * np.power(w22, 2))
    # m6_o = (lamb) * (np.power(mu1, 6) + 15 * np.power(mu1, 4) * w11 + 45 * np.power(mu1, 2) * np.power(w11,
    #                                                                                                  2) + 15 * np.power(
    #     w11, 3)) \
    #      + (1 - lamb) * (np.power(mu2, 6) + 15 * np.power(mu2, 4) * w22 + 45 * np.power(mu2, 2) * np.power(w22,
    #                                                                                                        2) + 15 * np.power(
    #     w22, 3))

    m1 = lamb * mu1 + (1 - lamb) * mu2
    m2 = (lamb) * (np.power(mu1, 2) + np.power(sigma1, 2)) \
         + (1 - lamb) * (np.power(mu2, 2) + np.power(sigma2, 2))
    m3 = (lamb) * (np.power(mu1, 3) + 3 * mu1 * np.power(sigma1, 2)) \
         + (1 - lamb) * (np.power(mu2, 3) + 3 * mu2 * np.power(sigma2, 2))
    m4 = (lamb) *           (np.power(mu1, 4) +
                           6 * np.power(mu1, 2) * np.power(sigma1, 2) +
                           3 * np.power(sigma1, 4)) \
         + (1 - lamb) *     (np.power(mu2, 4) +
                             6 * np.power(mu2, 2) * np.power(sigma2, 2) +
                             3 * np.power(sigma2, 4))

    m5 = (lamb) * (np.power(mu1, 5) +
                   10 * np.power(mu1, 3) * np.power(sigma1, 2) +
                   15 * mu1 * np.power(sigma1, 4)) \
         + (1 - lamb) * (np.power(mu2, 5) +
                         10 * np.power(mu2, 3) * np.power(sigma2, 2) +
                         15 * mu2 * np.power(sigma2, 4))
    m6 = (lamb) * (np.power(mu1, 6) +
                   15 * np.power(mu1, 4) * np.power(sigma1, 2) +
                   45 * np.power(mu1, 2) * np.power(  sigma1, 4) +
                   15 * np.power(sigma1, 6)) \
         + (1 - lamb) * (np.power(mu2, 6) +
                         15 * np.power(mu2, 4) * np.power(sigma2, 2) +
                         45 * np.power(mu2, 2) * np.power( sigma2, 4) +
                         15 * np.power(sigma2, 6))
    moments = np.array([m0, m1, m2, m3, m4, m5, m6])
    return moments


def MoG_Cumulants(moments):
    m0, m1, m2, m3, m4, m5, m6 = moments
    k1 = m1
    k2 = m2 - np.power(m1, 2)
    k3 = m3 - 3 * m1 * m2 + 2 * np.power(m1, 3)
    k4 = m4 - 4 * m1 * m3 - 3 * np.power(m2, 2) + 12 * np.power(m1, 2) * m2 - 6 * np.power(m1, 4)
    k5 = m5 - 5 * m1 * m4 - 10 * m2 * m3 + 20 * np.power(m1, 2) * m3 + 30 * m1 * np.power(m2, 2) - 60 * np.power(m1,
                                                                                                                 3) * m2 + 24 * np.power(
        m1, 5)
    k6 = m6 - 6 * m1 * m5 - 15 * m2 * m4 + 30 * np.power(m1, 2) * m4 - 10 * np.power(m3,
                                                                                     2) + 120 * m1 * m2 * m3 - 120 * np.power(
        m1, 3) * m3 + 30 * np.power(m2, 3) - 270 * np.power(m1, 2) * np.power(m2, 2) + 360 * np.power(m1,
                                                                                                      4) * m2 - 120 * np.power(
        m1, 6)

    return k1, k2, k3, k4, k5, k6


def MoG_Get_MoG(moments):
    k1, k2, k3, k4, k5, k6 = MoG_Cumulants(moments)

    poly = lambda p: (8) * np.power(p, 9) + \
                     (28 * k4) * np.power(p, 7) + \
                     (12 * np.power(k3, 2)) * np.power(p, 6) + \
                     (24 * k3 * k5 + 30 * np.power(k4, 2)) * np.power(p, 5) + \
                     (148 * np.power(k3, 2) * k4 - 6 * np.power(k5, 2)) * np.power(p, 4) + \
                     (96 * np.power(k3, 4) + 9 * np.power(k4, 3) - 36 * k3 * k4 * k5) * np.power(p, 3) + \
                     (-21 * np.power(k3, 2) * np.power(k4, 2) - 24 * np.power(k3, 3) * k5) * np.power(p, 2) + \
                     (-32 * np.power(k3, 4) * k4) * np.power(p, 1) + \
                     (-8 * np.power(k3, 6))

    pol = np.array([(8), 0, (28 * k4), (12 * np.power(k3, 2)), (24 * k3 * k5 + 30 * np.power(k4, 2)),
                    (148 * np.power(k3, 2) * k4 - 6 * np.power(k5, 2)),
                    (96 * np.power(k3, 4) + 9 * np.power(k4, 3) - 36 * k3 * k4 * k5),
                    (-21 * np.power(k3, 2) * np.power(k4, 2) - 24 * np.power(k3, 3) * k5), (-32 * np.power(k3, 4) * k4),
                    (-8 * np.power(k3, 6))])
    p = np.roots(pol)  # find roots of polynomial
    p = p[~np.iscomplex(p)]  # only real roots
    p = p.real
    p = p[p <= 0]

    for this_p in p:
        print(this_p / mu1)
        if (4 * np.power(this_p, 3) * k3 - 4 * np.power(k3, 3) - 6 * this_p * k3 * k4 - 2 * np.power(this_p,
                                                                                                     2) * k5) != 0:
            this_s = - (4 * np.power(this_p, 5) + 14 * np.power(this_p, 2) * np.power(k3, 2) + 8 * np.power(this_p,
                                                                                                            3) * k4 + np.power(
                k3, 2) * k4 + 3 * this_p * np.power(k4, 2) - 2 * this_p * k3 * k5) / (
                                 4 * np.power(this_p, 3) * k3 - 4 * np.power(k3,
                                                                             3) - 6 * this_p * k3 * k4 - 2 * np.power(
                             this_p, 2) * k5)
            poly = np.array([1, -this_s, this_p])
            mu1, mu2 = np.roots(poly)
            print("mu1: ", mu1, " mu2: ", mu2)
            lamb = -mu2 / (mu1 - mu2)
            print("Lambda: ", lamb)

            R1 = -k2 - this_p
            R1_tilde = R1 * (mu1 - mu2)
            R2 = (this_p * this_p - k3) / (3 * this_p)
            R2_tilde = R2 * (mu1 - mu2)

            sigma1 = np.sqrt((R1_tilde - mu1 * R2_tilde) / (mu1 - mu2))
            sigma2 = np.sqrt(R2_tilde + np.power(sigma1, 2))
            print("sigma1: ", sigma1, " sigma2: ", sigma2)
