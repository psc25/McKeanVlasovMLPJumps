import numpy as np
import scipy.special as ssp
from McKV_MLP_Jump import MLP_model
import matplotlib.pyplot as plt
import time
import os

path = "vg"
cluster = True

if cluster:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type = int, required = True)
    parser.add_argument("--nm1", type = int, required = True)
    parser.add_argument("--nm2", type = int, required = True)
    parser.add_argument("--r1", type = int, required = True)
    parser.add_argument("--r2", type = int, required = True)
    args = parser.parse_args()
    d = args.d
    mns = list(range(args.nm1, args.nm2+1))
    runs = list(range(args.r1, args.r2+1))
else:
    d = 10
    mns = list(range(1, 5))
    runs = list(range(1, 11))
    
T = 1.0
print("======================================================================")

np.random.seed(0)
def sym_beta(size, supp, alpha = 2.0, dtype = np.float32):
    return (supp * (2.0 * np.random.beta(alpha, alpha, size=size) - 1.0)).astype(dtype)

a = np.random.uniform(size = [1, d], low = -1.0/np.sqrt(d), high = 1.0/np.sqrt(d)).astype(np.float32)
A1 = np.random.uniform(size = [d, d], low = -1.0/d, high = 1.0/d).astype(np.float32)
A1T = np.transpose(A1)
A2 = np.random.uniform(size = [d, d], low = -1.0/d, high = 1.0/d).astype(np.float32)
A2T = np.transpose(A2)
C = np.random.uniform(size = [d, d, d], low = -1.0/np.power(d, 1.5), high = 1.0/np.power(d, 1.5)).astype(np.float32)
C2 = np.reshape(C, [d**2, d])
alpha = 1.0
kappa = 0.1
delta = 0.02
delta2 = 0.001

def mu(x, y):
    return a + np.matmul(x, A1T) + np.matmul(y, A2T)

def sigma(x, y):
    return np.zeros([d, d])

def eta(x, y, z):
    Cx = np.reshape(np.matmul(C2, x.ravel()), [d, d])
    return np.matmul(z, Cx)

c = (0.2168+0.932*d/2.0)/(0.392+d/2.0)
gamma = 2.0*np.power(d, c)/(1.0+np.power(d, c))
lambd = gamma*np.sqrt(np.pi)*np.exp(ssp.gammaln(d/2.0+0.5)-ssp.gammaln(d/2.0))/ssp.gamma(1.0/gamma)

# Approximate K_{d/2}(x) by the function in https://arxiv.org/abs/2303.13400
def CDF(x):
    exp1 = ssp.exp1(np.power(np.sqrt(2*alpha/kappa)*x/lambd, gamma))
    return 2*alpha*exp1/gamma

nu_lam = CDF(delta)
nu_lam2 = CDF(delta2)

# Approximate the inverse of E1(x) by the following function
# see also https://mathematica.stackexchange.com/questions/251068/asymptotic-inversion-of-expintegralei-function
eulermasc = 0.5772156649
def invE1(x):
    y1 = -np.log(x)-np.log(-np.log(x))-(np.log(-np.log(x))-1.0)/np.log(x)
    y2 = np.exp(-(x+eulermasc))+np.exp(-2.0*(x+eulermasc))+1.25*np.exp(-3.0*(x+eulermasc))
    y1 = np.expand_dims((x < 0.2043338275)*y1, -1)
    y2 = np.expand_dims((x >= 0.2043338275)*y2, -1)
    y = np.concatenate([y1, y2], axis = -1)
    return np.nansum(y, axis = -1)

def invCDF(x):
    y = np.power(invE1(nu_lam*gamma*x/(2.0*alpha)), 1/gamma)
    return np.sqrt(kappa/(2.0*alpha))*lambd*y

def invCDF2(x):
    y = np.power(invE1(nu_lam2*gamma*x/(2.0*alpha)), 1/gamma)
    return np.sqrt(kappa/(2.0*alpha))*lambd*y

def jump_rvs(M):
    Y = np.random.normal(size = [M, d])
    V = Y/np.linalg.norm(Y, axis = -1, keepdims = True)
    U = np.random.uniform(low = 0.0, high = 1.0, size = [M, 1])
    R = invCDF(U)
    Z = R*V
    return Z.astype(np.float32)

def coupled_jumps(M):
    Y = np.random.normal(size = [M, d])
    V = Y/np.linalg.norm(Y, axis = -1, keepdims = True)
    U = np.random.uniform(low = 0.0, high = 1.0, size = [M, 1])
    R = invCDF2(U)
    Z = R*V
    return Z, U

mu_cost = 4*d**2
sigma_cost = 0
eta_cost = lambda q: 2*d**3 - d**2 + q*(2*d**2 - d)
for mn in mns:
    K = np.power(mn, mn)
    k = int(np.ceil(500/K))
    K1 = k*K
    tt = np.reshape(np.linspace(0.0, T, K+1), [-1, 1])
    dt = T/K
    dt1 = T/K1
    xi = 10.0*np.ones([1, d], dtype = np.float32)
    for r in runs:
        np.random.seed(r)
        dW1 = np.random.normal(size = [K1, d], scale = np.sqrt(T/K1))
        W1 = np.cumsum(np.concatenate([np.zeros([1, d]), dW1], axis = 0), axis = 0)
        dW = np.add.reduceat(dW1, np.arange(stop = K1, step = k))
        W = np.cumsum(np.concatenate([np.zeros([1, d]), dW], axis = 0), axis = 0)
        
        dN1 = np.random.poisson(size=K1, lam=nu_lam2 * dt1)
        N1 = np.cumsum(np.insert(dN1, 0, 0))
        Z1, U1 = coupled_jumps(N1[-1])
        keep = (U1[:, 0] < nu_lam/nu_lam2)
        Z = Z1[keep]
        prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(keep, dtype=np.int64)])
        dN1_delta = prefix[N1[1:]] - prefix[N1[:-1]]
        dN = np.add.reduceat(dN1_delta, np.arange(stop=K1, step=k))
        N = np.cumsum(np.insert(dN, 0, 0))
        
        beg = time.time()
        MLP = MLP_model(d, mn, mn, K, T, mu, sigma, eta, jump_rvs, nu_lam, mu_cost, sigma_cost, eta_cost)
        X_mlp = MLP.compute(xi, W, N, Z)
        
        cst = MLP.cost
        end = time.time()
        
        print("MLP performed for d = " + str(d) + ", m = " + str(mn) + ", r = " + str(r) + ", in " + str(np.round(end-beg, 1)) + "s")
        
        m = np.zeros([K1+1, d])
        m[0] = xi[0]
        X_tru = np.zeros([K1+1, d])
        X_tru[0] = xi[0]
        for t in range(1, K1+1):
            m[t:t+1] = m[t-1:t] + (a + np.matmul(m[t-1:t], A1T + A2T))*dt1
            drift = mu(X_tru[t-1:t], m[t-1:t])*dt1
            diffu = np.matmul(dW1[t-1:t], sigma(X_tru[t-1:t], m[t-1:t]).T)
            jumps = np.sum(eta(X_tru[t-1:t], m[t-1:t], Z1[N1[t-1]:N1[t]]), axis = 0)
            # the compensator is in this model analytically zero!
            X_tru[t:t+1] = X_tru[t-1:t] + drift + diffu + jumps
            
        np.savetxt(os.path.join(path, "mlp_" + str(d) + "_" + str(mn) + "_" + str(r) + ".csv"), X_mlp)
        np.savetxt(os.path.join(path, "tru_" + str(d) + "_" + str(mn) + "_" + str(r) + ".csv"), X_tru[::k])
        np.savetxt(os.path.join(path, "tms_" + str(d) + "_" + str(mn) + "_" + str(r) + ".csv"), [end-beg])
        np.savetxt(os.path.join(path, "cst_" + str(d) + "_" + str(mn) + "_" + str(r) + ".csv"), [cst])
        
        fig = plt.figure()
        plt.plot(X_mlp, linestyle = "dotted")
        plt.plot(X_tru[::k], linestyle = "solid")
        plt.savefig(os.path.join(path, "sim_" + str(d) + "_" + str(mn) + "_" + str(r) + ".png"), dpi = 500)
        plt.show()
        plt.close()
        
print("======================================================================")
print("MLP solutions saved")