'''Testing file, let's craft something!'''

import numpy as np
from mcup import VirtualExperiment as VE
from mcup import LeastSquares as LSQ
from mcup import PropagatorErrorEstimator as PEE
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from tqdm import tqdm
import pickle

def pdf(x, mu, sigma):
    x = float(x - mu) / sigma
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma

def fun1(x, a, b):
    return a*x+b

def fun2(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d

def fun3(x, a, b):
    return a/(x-b)

def fun4(x, a, b):
    return a*np.exp(b*x)

def fun5(x, a, b, c):
    return a*np.sin(b*(x-c))

def fun6(x, a, b):
    return np.exp(-b*x)*np.cos(a*x)

N = 30
params = np.array([1, 10])
w_0 = np.array([1, 1])
N_runs = 5*10**3
x_error_range = np.linspace(0, 1, N)
y_error_range = np.linspace(0, 1, N)
mud_dist = np.zeros((N, N, len(params)))
std_dist = np.zeros((N, N, len(params)))

lsq = LSQ("scipy")
lsq.set_params(method='lm', loss='linear', max_nfev=100)
np.linspace(0.05, 5, N)
for xe_ind in tqdm(range(N)):
    for ye_ind in tqdm(range(N)):
        ve = VE(
            fun1, x_bounds=[0, 10], sample_num=10, 
            x_error=x_error_range[xe_ind], y_error=y_error_range[ye_ind], params=params)
        
        x, y, x_errs, y_errs = ve.measure()
        pee = PEE(x, y, x_errs, y_errs, w_0)
        w_distribution, var_distribution = pee.run(lsq, fun1, sample_num=N_runs, n_thread=10)
        for w_i in range(len(params)):
            mu, std = norm.fit(w_distribution[:,w_i])
            mud_dist[xe_ind][ye_ind][w_i] = abs(params[w_i] - mu)
            std_dist[xe_ind][ye_ind][w_i] = std

filename = "linear_1_10"
x_label = map(lambda x: str(round(x, 1)), [x for x in x_error_range])
x_label = list(x_label)
y_label = map(lambda x: str(round(x, 1)), [x for x in y_error_range])
y_label = list(y_label)
for i in range(len(params)):
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.matshow(mud_dist[:,:,i], interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(x_label)), x_label, rotation='vertical') 
    plt.yticks(range(len(y_label)), y_label) 
    plt.savefig("img/"+filename+"_"+str(i)+"_std_mat.png", dpi=600)
for i in range(len(params)):
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.matshow(std_dist[:,:,i], interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(x_label)), x_label, rotation='vertical') 
    plt.yticks(range(len(y_label)), y_label) 
    plt.savefig("img/"+filename+"_"+str(i)+"_mu_mat.png", dpi=600)

with open("pkl/"+filename+".pkl", "wb") as f:
    data = [mud_dist, std_dist]
    pickle.dump(data, f)