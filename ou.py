import numpy as np
from McKV_MLP_Jump import MLP_model
import matplotlib.pyplot as plt
import time
import os

path = "ou"
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
a = np.random.uniform(size = [1, d], low = -0.5/np.sqrt(d), high = 0.5/np.sqrt(d)).astype(np.float32)
A1 = np.random.uniform(size = [d, d], low = -0.25/d, high = 0.25/d).astype(np.float32)
A1T = np.transpose(A1)
A2 = np.random.uniform(size = [d, d], low = -0.25/d, high = 0.25/d).astype(np.float32)
A2T = np.transpose(A2)
b = np.random.uniform(size = [d, d], low = -0.5/d, high = 0.5/d).astype(np.float32)
bT = np.transpose(b)
B = np.random.uniform(size = [d, d, d], low = -0.25/np.power(d, 1.5), high = 0.25/np.power(d, 1.5)).astype(np.float32)
BT = np.transpose(B, axes = [0, 2, 1])
C = np.random.uniform(size = [d, d, d], low = -0.25/np.power(d, 1.5), high = 0.25/np.power(d, 1.5)).astype(np.float32)
C2 = np.reshape(C, [d**2, d])
nu_lam = 5.0

def mu(x, y):
    return a + np.matmul(x, A1T) + np.matmul(y, A2T)

def sigma(x, y):
    return bT + np.matmul(x.ravel(), BT)

def eta(x, y, z):
    Cx = np.reshape(np.matmul(C2, x.ravel()), [d, d])
    return np.matmul(z, Cx)

def jump_rvs(M):
    return np.random.normal(size = [M, d], scale = 1.0)

mu_cost = 4*d**2
sigma_cost = 2*d**3
eta_cost = lambda q: 2*d**3 - d**2 + q*(2*d**2 - d)
for mn in mns:
    K = np.power(mn, mn)
    k = int(np.ceil(500/K))
    K1 = k*K
    tt = np.reshape(np.linspace(0.0, T, K+1), [-1, 1])
    dt = T/K
    dt1 = T/K1
    xi = 20.0*np.ones([1, d], dtype = np.float32)
    for r in runs:
        np.random.seed(r)
        dW1 = np.random.normal(size = [K1, d], scale = np.sqrt(T/K1))
        W1 = np.cumsum(np.concatenate([np.zeros([1, d]), dW1], axis = 0), axis = 0)
        dW = np.add.reduceat(dW1, np.arange(stop = K1, step = k))
        W = np.cumsum(np.concatenate([np.zeros([1, d]), dW], axis = 0), axis = 0)
        
        dN1 = np.random.poisson(size = K1, lam = nu_lam*dt1)
        N1 = np.cumsum(np.insert(dN1, 0, 0))
        Z = jump_rvs(N1[-1])
        dN = np.add.reduceat(dN1, np.arange(stop = K1, step = k))
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
            jumps = np.sum(eta(X_tru[t-1:t], m[t-1:t], Z[N1[t-1]:N1[t]]), axis = 0)
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