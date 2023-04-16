# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 11:22:24 2022

@author: Pranav
"""

import numpy as np
from numpy import sqrt
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

def project(q, z, Q, grad, dim, nmax=0, tol=0.01):
    a = np.zeros(dim)
    i = 0
    if nmax == 0:
        nmax = 3 * dim
    while i == 0 or np.linalg.norm(q_arg) > tol:
        arg = z + Q @ a
        q_arg = q(arg)
        da = -np.linalg.pinv(grad(arg).T @ Q) @ q_arg
        a = a + da
        i += 1
        if i > nmax:
            return a, False
    # print(np.linalg.norm(q(z + Q @ a)))
    return a, np.linalg.norm(q(z + Q @ a)) < tol

def mcmc_manifold(N, d, m, grad, q, x0, sigma, ineq_constraints=None, check=None):
    da = d - m
    X = np.zeros((N + 1, d))
    X[0] = x0
    accepted = 0
    cov = np.eye(da) * sigma
    for i in range(N):
        print(i)
        X[i + 1] = X[i]
        Gx = grad(X[i])
        tmp = np.linalg.qr(Gx, mode='complete')
        qrx = np.linalg.qr(Gx, mode='complete')[0][:, m:]
        t = multivariate_normal.rvs(cov=cov)
        if not isinstance(t, np.ndarray):
            t = [t]
        v = qrx @ t
        # tt = Gx.reshape(1, -1)[0]
        ttt = [v.dot(Gx[:, i]) for i in range(m)]
        a, flag = project(q, X[i] + v, Gx, grad, m)
        if not flag:
            continue
        w = Gx @ a
        Y = X[i] + v + w
        if ineq_constraints is not None and not ineq_constraints(Y):
            continue
        Gy = grad(Y)
        qry = np.linalg.qr(Gy, mode='complete')[0][:, m:]
        v_ = qry @ qry.T @ (X[i] - Y)
        alpha = min(1, np.exp(-(np.linalg.norm(v_) ** 2 - np.linalg.norm(v) ** 2) / 2 / sigma ** 2))
        U = np.random.uniform()
        if U > alpha:
            continue
        reversebility_check, flag = project(q, Y + v_, Gy, grad, m)
        if not flag:
            continue
        if check is not None and not check(Y):
            continue
        X[i + 1] = Y
        accepted += 1

    return X, accepted / N

np.random.seed(10)


def G_col(X, k, l):
    # Assume X is dxd matrix
    d = X.shape[0]
    col = np.zeros(d * d)
    if k == l:
        for j in range(d):
            col[k * d + j] = 2 * X[k, j]
    else:
        for j in range(d):
            col[k * d + j] = X[l, j]
            col[l * d + j] = X[k, j]
    return col


def G(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    m = int(d * (d + 1) / 2)
    res = np.zeros((d * d, m))
    for k in range(d):
        for l in range(k, d):
            res[:, d * k + l - int(k * (k + 1) / 2)] = G_col(X, k, l)
    return res


def q_col(X, k, l):
    # Assume X is dxd matrix
    if k == l:
        return np.linalg.norm(X[k, :]) ** 2 - 1
    else:
        return X[k, :].dot(X[l, :])


def q(X):
    d = int(sqrt(X.shape[0]))
    X = X.reshape(d, d)
    m = int(d * (d + 1) / 2)
    res = np.zeros(m)
    for k in range(d):
        for l in range(k, d):
            res[d * k + l - int(k * (k + 1) / 2)] = q_col(X, k, l)
    return res


def is_so(X):
    d = int(sqrt(X.shape[0]))
    return np.linalg.det(X.reshape(d, d)) > 0


N = 100000
d = 2
# sigma = 4.5
sigma = 0.27
x0 = np.eye(d)
# x0 = np.random.normal(size=(d, d))
# np.random.shuffle(x0)
# print(G(x0.flatten()))
X, prob = mcmc_manifold(N, d * d, int(d * (d + 1) / 2), G, q, x0.flatten(), sigma, is_so)
print(prob)
X = X.reshape(N + 1, d, d)
# traces = np.array([np.trace(Xi) for Xi in X])
# plt.subplots()
# plt.hist(traces, density=True)



phi = np.array([np.arctan2(Xi[0, 0], Xi[1, 0]) for Xi in X])
plt.subplots()
plt.hist(phi, density=True)
plt.show()





