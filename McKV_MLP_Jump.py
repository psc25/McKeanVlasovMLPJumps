import numpy as np

class MLP_model():
    def __init__(self, d, m, n, K, T, mu, sigma, eta, jump_rvs, nu_lam, mu_cost, sigma_cost, eta_cost, dtype = np.float32):
        self.d = d
        self.m = m
        self.n = n
        self.K = K
        self.T = T
        self.dt = T/K
        self.tt = np.reshape(np.linspace(0.0, T, K+1), [-1, 1])
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.nu_lam = nu_lam
        self.nu_ceil = int(np.ceil(nu_lam))
        self.jump_rvs = jump_rvs
        self.mu_cost = int(mu_cost)
        self.sigma_cost = int(sigma_cost)
        self.eta_cost = eta_cost
        self.dtype = dtype
        self.cost = np.array([0], dtype = np.int64)
        
    def simulate_path(self):
        dW = np.random.normal(size = [self.K, self.d], scale = np.sqrt(self.dt)).astype(dtype = self.dtype)
        W = np.cumsum(np.concatenate([np.zeros([1, self.d]), dW], axis = 0), axis = 0)
        N = np.cumsum(np.insert(np.random.poisson(size = self.K, lam = self.nu_lam*self.dt), 0, 0))
        Z = self.jump_rvs(N[-1]).astype(dtype = self.dtype)
        self.cost += self.K*(self.d + 3) + N[-1]
        return W, N, Z
    
    def simulate_compensator(self):
        V = self.jump_rvs(self.K*self.nu_ceil).astype(dtype = self.dtype).reshape([self.K, self.nu_ceil, self.d])
        self.cost += self.K*self.nu_ceil
        return V
    
    def recursion(self, xi, m, n, W, N, Z):
        if n <= 0:
            return np.zeros([self.K+1, self.d], dtype = self.dtype)
        
        X = np.tile(xi, [self.K+1, 1])        
        for l in range(n):
            X1 = self.recursion(xi, m, l, W, N, Z)
            X2 = self.recursion(xi, m, l-1, W, N, Z)
            
            mnl = np.power(self.m, n-l)
            for s in range(mnl):
                W_, N_, Z_ = self.simulate_path()
                V = self.simulate_compensator()
                
                X1_ = self.recursion(xi, m, l, W_, N_, Z_)
                X2_ = self.recursion(xi, m, l-1, W_, N_, Z_)
                
                IW = np.zeros([self.K+1, self.d], dtype = self.dtype)
                IN = np.zeros([self.K+1, self.d], dtype = self.dtype)
                IC = np.zeros([self.K+1, self.d], dtype = self.dtype)
                for t in range(1, self.K+1):
                    IW[t:t+1] = IW[t-1:t] + np.matmul(W[t] - W[t-1], (self.sigma(X1[t-1:t], X1_[t-1:t]) - (l > 0)*self.sigma(X2[t-1:t], X2_[t-1:t])).T)/mnl
                    self.cost += 2*self.sigma_cost + 4*self.d**2 + 2*self.d
                    IN[t:t+1] = IN[t-1:t] + np.sum(self.eta(X1[t-1:t], X1_[t-1:t], Z[N[t-1]:N[t]]) - (l > 0)*self.eta(X2[t-1:t], X2_[t-1:t], Z[N[t-1]:N[t]]), axis = 0)/mnl
                    q = N[t] - N[t-1]
                    self.cost += 2*self.eta_cost(q) + 2*q*self.d + max(q-1, 0)*self.d + 2*self.d
                    IC[t:t+1] = IC[t-1:t] + self.nu_lam*np.mean(self.eta(X1[t-1:t], X1_[t-1:t], V[t-1]) - (l > 0)*self.eta(X2[t-1:t], X2_[t-1:t], V[t-1]), axis = 0, keepdims = True)*self.dt/mnl
                    self.cost += 2*self.eta_cost(self.nu_ceil) + 3*self.nu_ceil*self.d + 4*self.d
                
                X = X + IW + IN - IC
                self.cost += 3*(self.K+1)*self.d
                
                u = np.random.uniform()
                self.cost = self.cost + 1
                for t in range(1, self.K+1):
                    r = max(int(np.floor(t*u))-1, 0)
                    X[t:t+1] = X[t:t+1] + self.tt[t]*(self.mu(X1[r:r+1], X1_[r:r+1]) - (l > 0)*self.mu(X2[r:r+1], X2_[r:r+1]))/mnl
                    self.cost += 2*self.mu_cost + 5*self.d + 1
                
        return X
    
    def compute(self, xi, W, N, Z):
        return self.recursion(xi, self.m, self.n, W, N, Z)