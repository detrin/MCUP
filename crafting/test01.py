'''Testing file, let's craft something!'''

import numpy as np
from mcup import VirtualExperiment as VE
from mcup import LeastSquares as LSQ
from mcup import PropagatorErrorEstimator as PEE
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm

def pdf(x, mu, sigma):
    x = float(x - mu) / sigma
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma

def fun(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d

ve = VE(fun, x_bounds=[-1, 1], sample_num=100, x_error=0.5, y_error=0, params=[1, -5, 2, 5])
lsq = LSQ("scipy")
lsq.set_params(method='lm', loss='linear', max_nfev=1000)
w_0 = [1, 1, 1, 1]

N_runs = 10**4
timeFlag = time.time()

x, y, x_errs, y_errs = ve.measure()
pee = PEE(x, y, x_errs, y_errs, w_0)
w_distribution, var_distribution = pee.run(lsq, fun, sample_num=N_runs, n_thread=10)

timeTotal = time.time()-timeFlag
print(timeTotal, "s total, ", timeTotal/N_runs, "s per run")

bin_num = 100
fig, axes = plt.subplots(2, len(w_0), figsize=(3*len(w_0),4))
fig.tight_layout()
for w_i in range(len(w_0)):
    axes[0, w_i].hist(w_distribution[:,w_i], bins=bin_num, density=True, alpha=0.6, color='g')
    xmin, xmax = min(w_distribution[:,w_i]), max(w_distribution[:,w_i])
    x = np.linspace(xmin, xmax, 10*bin_num)
    mu, std = norm.fit(w_distribution[:,w_i])
    print(mu, std)
    p = [pdf(val, mu, std) for val in x]
    axes[0, w_i].plot(x, p, 'k', linewidth=1)
    title_text = "i="+str(w_i)+", mu=%.2f, std=%.2f" % (mu, std)
    axes[0, w_i].set_title(title_text, fontsize=7)
    axes[0, w_i].set_xlim(mu-4*std, mu+4*std)

for w_i in range(len(w_0)):
    axes[1, w_i].hist(var_distribution[:,w_i], bins=bin_num, density=True, alpha=0.6, color='g')
    xmin, xmax = min(var_distribution[:,w_i]), max(var_distribution[:,w_i])
    x = np.linspace(xmin, xmax, 10*bin_num)
    mu, std = norm.fit(var_distribution[:,w_i])
    p = [pdf(val, mu, std) for val in x]
    axes[1, w_i].plot(x, p, 'k', linewidth=1)
    title_text = "i="+str(w_i)+", mu=%.2f, std=%.2f" % (mu, std)
    axes[1, w_i].set_title(title_text, fontsize=7)
    axes[1, w_i].set_xlim(mu-4*std, mu+4*std)

plt.show()