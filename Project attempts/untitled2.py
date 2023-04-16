# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:17:35 2022

@author: Pranav
"""

import numpy as np
from numpy import sqrt
from matplotlib import pyplot as plt
np.random.seed(10)
import numpy as np
from numpy import sqrt
from scipy.stats import multivariate_normal


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

def q(X):
    [x, y, z] = X
    return np.array([x**2 + y**2 + z**2 - 1, (x-0.5)**2 + y**2 + z**2 - 1])


def G(X):
    [x, y, z] = X
    return np.array([[2*x, 2*x - 1], [2*y, 2*y], [2*z, 2*z]])


def h(X):
    [x, y, z] = X
    return z > 0 and x > 0


N = 1000
# a = (3. - sqrt(23.)) / 8.
x0 = np.array([0.25, sqrt(15.) / 4., 0])
sigma = 0.01
X, prob = mcmc_manifold(N, 3, 2, G, q, x0, sigma)
print(prob)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, label='MCMC')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.1, 1.1)
ax.legend()
plt.show()