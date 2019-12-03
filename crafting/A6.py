'''Yay, heading for A6!'''

import numpy as np
from mcup import LeastSquares as LSQ
from mcup import PropagatorErrorEstimator as PEE
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def pdf(x, mu, sigma):
    x = float(x - mu) / sigma
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma

def fun(x, a, b):
    return a+b*x**0.5

with open("A6.dat", "r") as f:
    buffer = []
    for row in f:
        row = row.replace("\n", "")
        if " " in row:
            row = row.split(" ")
        if "\t" in row:
            row = row.split("\t")
        row = filter(lambda x: x!="", row)
        row = map(float, row)
        row = list(row)
        if len(row) > 0:
            buffer.append(row)

data = np.zeros((len(buffer), 4), dtype='float64')
for row_i in range(len(buffer)):
    for j in [0, 1, 2, 3]:
        data[row_i][j] = buffer[row_i][j]
print(data)

x, x_errs, y, y_errs = data[:,0], data[:,1], data[:,2], data[:,3]
w_0 = [20, 7]
par_names = ['a', 'b']
N_run = 10**5

lsq = LSQ("scipy")
lsq.set_params(method='trf', loss='linear')

timeFlag = time.time()
pee = PEE(x, y, x_errs, y_errs, w_0)
w_distribution, var_distribution = pee.run(lsq, fun, sample_num=N_run, 
    n_thread=10, method='normal')
timeTotal = time.time()-timeFlag
print(timeTotal, "s total, ")


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
    title_text = par_names[w_i]+r", $\mu$=%.2f, $\sigma$=%.2f" % (mu, std)
    axes[0, w_i].set_title(title_text, fontsize=10)
    # axes[0, w_i].set_xlim(mu-4*std, mu+4*std)
    axes[0, w_i].set_yticks([], []) 
    axes[0, w_i].set_ylabel("N") 

for w_i in range(len(w_0)):
    axes[1, w_i].hist(var_distribution[:,w_i], bins=bin_num, density=True, alpha=0.6, color='g')
    xmin, xmax = min(var_distribution[:,w_i]), max(var_distribution[:,w_i])
    x = np.linspace(xmin, xmax, 10*bin_num)
    mu, std = norm.fit(var_distribution[:,w_i])
    p = [pdf(val, mu, std) for val in x]
    axes[1, w_i].plot(x, p, 'k', linewidth=1)
    title_text = r"$\sigma$("+par_names[w_i]+r"), $\mu$=%.2f, $\sigma$=%.2f" % (mu, std)
    axes[1, w_i].set_title(title_text, fontsize=10)
    # axes[1, w_i].yaxis.set_major_formatter(plt.NullFormatter()) 
    axes[1, w_i].set_yticks([], [])
    axes[1, w_i].set_ylabel("N") 


plt.savefig("A6_stat.png", size=(4, 4), dpi=600)
plt.show()