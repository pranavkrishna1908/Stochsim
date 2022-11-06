from scipy.stats import burr12, norm
import numpy as np
import matplotlib.pyplot as plt

def estimate(samples, x, delta = 0.01):
    n = len(samples)
    return sum(norm.pdf(samples,loc = x, scale = delta))/(n*delta)

def ecdf(x, n):
    ecdf = np.zeros(n)
    for i in range(n):
        ecdf[i] = sum(x <= (i+1)/n)   
    return ecdf

def estimate_sim(samples, x):
    n = len(samples)
    delta = n**(-0.2)
    ans = np.repeat(0., len(x))
    for i in range(len(x)):
        ans[i] = estimate(samples, x[i], delta)
    return ans

n_s = np.array([100,10000])
x_s = np.arange(0,5,0.1)
plt.figure(0)
for i in range(len(n_s)):
    samples = burr12.rvs(2,4,size = n_s[i])
    y_s = estimate_sim(samples, x_s)
    plt.plot(x_s,y_s)
plt.plot(x_s, burr12.pdf(x_s,2,4))







#pdf is incorrect in scale, shape looks correct




plt.figure(1)
num_samples = 200
values = np.zeros((len(n_s),num_samples))
for i in range(len(n_s)):
    delta = (n_s[i])**(-0.2)
    samples = burr12.rvs(2,4, size = n_s[i])
    for j in range(num_samples):
        temp = np.random.randint(n_s[i])
        values[i,j] = norm.rvs(loc = samples[temp], scale = delta)
    plt.plot(ecdf(values[i,:], int(n_s[i]/ 5)))
