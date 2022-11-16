import numpy as np
import torch as th
from scipy.special import softmax
from torch.optim import optim


class DeepLSR(th.nn.Module):
    def __init__(self, n, nnet, sigma_function, rho=1):
        super().__init__()
        self.n = n
        self.nnet = nnet
        if not hasattr(nnet, "K") or not hasattr(nnet, "d_item") or not hasattr(nnet, "d_user"):
            print("nnet should have a K attribute (number of mixtures), d_item and d_user")
        self.K = nnet.K
        self.d_item = nnet.d_item
        self.d_user = nnet.d_user
        self.nu = 1
        self.rho = rho
        self.sigma_function = sigma_function
    
    def fit(self, rankings, X_item, X_user, max_iters=100):
        # Assume that rankings is a list of (user, ranking) tuples
        S = self.construct_choice_tensor(rankings)
        
        # Obtain some initial estimate
        
        # Initialize pik
        for _ in range(max_iters):
            # Update pi
            pi = self.update_pi()
            
            # Update y
            y = self.update_y()
            
            # Update neural net
            self.update_neural_net()
            
            # Compute sample weights
            alpha = self.estimate_alpha(rankings, self.nnet)
        
        return
    
    def update_pi(self, choice_tensor, pi, tilde_pi, sample_weights, y, rho, max_ilsr_iters=100, eps=1e-4):
        
        for k in range(self.K):
            # Update each pik separately
            for _ in range(max_ilsr_iters):
                pi_k = pi[:, k]
                tilde_pi_k = tilde_pi[:, k]
                sigma_k = self.sigma_function(pi_k, tilde_pi_k)
                yk = y[:, k]
                M, d = self.construct_markov_chain(choice_tensor, pi_k, yk, rho, sigma_k, sample_weights[:, k])
                pi_knext = self.compute_stationary_distribution(M, np.transpose(pi_k[:, np.newaxis]))
                pi_knext = pi_knext/d
                pi_knext = pi_knext/ np.sum(pi_knext)
                
                if np.linalg.norm(pi_knext - pi_k) < eps:
                    break
                pi[:, k] = pi_knext
        
        return pi

    def update_y(self, y, pi, pi_tilde, rho):
        # for k in range(self.K):
        #     y[:, k] = y[:, k] + rho * (pi[:, k] - pi_tilde[:, k])
        # return y
        y = y + rho * (pi - pi_tilde)
        return y
    
    def estimate_qz(self, rankings, ):
        
        
        return
        
    def update_neural_net(self, pi, nnet, X_item_user, y, loss_function, max_iters=100, lr=1e-3):
        optimizer = optim.SGD(nnet.parameters(), lr=lr)
        
        for _ in range(max_iters):
            optimizer.zero_grad()

            # For each item and in each cluster components, aggregate the utility
            tilde_theta = th.zeros(self.n, self.K)
            for i in range(self.n):
                Xi = X_item_user[i]
                out = nnet(Xi) # This should have shape (sample_size, 2K)
                
                for k in range(self.K): # Weighted average over alpha k * f(Xi, Ul)
                    hatbeta_ki = out[:, 2*k] * out[:, 2*k+1]/ th.sum(out[:, 2*k+1])
                    tilde_theta[i, k] = hatbeta_ki
                    
            # Apply softmax
            tilde_pi = th.nn.functional.softmax(tilde_theta, 0)
                    
            loss = 0.    
            # Then back propagate
            for k in range(self.K):
                loss += loss_function(pi[:, k], tilde_pi[:, k])
            loss += th.sum(y * (pi - tilde_pi))
            loss.backward()
            optimizer.step()
        
        return tilde_pi.detach().numpy()
    
    def construct_choice_tensor(self, ranked_data):
        # This returns a tensor S of size (n, m, n)
        # S where S[j, i, :] = {0,1}^l where S[j, i, l] = 1 if j is ranked ahead of i in pi_l
        S = np.zeros((self.n, self.n, len(ranked_data)))
        for l, (user_id, rank) in enumerate(ranked_data):
            for idj, j in enumerate(rank[:-1]):
                for i in rank[idj:]:
                    S[j, i, l] = 1
        return S

    def construct_markov_chain(self, S_choice_tensor, pi:np.array, y:np.array, rho:float, sigma:np.array, sample_weights:np.array=None):
        m = S_choice_tensor.shape[-1]
        if sample_weights is None:
            sample_weights = np.ones((m,))
                    
        # Accelerated construction
        temp = pi @ S_choice_tensor
        piSk = np.divide(1, temp, out=np.zeros_like(temp), where=temp!=0)        
        # Now we have to zero out all the S_choice_tensor[i, i, :]
        for i in range(self.n):
            S_choice_tensor[i, i, :] = np.zeros((m,))
        M = (np.transpose(sample_weights * S_choice_tensor, (1, 0, 2)) * piSk).sum(-1)
        np.fill_diagonal(M, 0)

        d = np.maximum(np.count_nonzero(M, 1))
        
        # This only takes O(n^2)
        for i in range(self.n):
            di = d[i]
            yi = y[i]
            sigmai = sigma[i]
            for j in np.nonzero(M[i, :])[0]:
                M[i, j] += (yi - rho * sigmai)/di # Pray that this is still > 0
        
        # Check everypair where if i flows into j, j should also have back flow
        # M = np.where(np.logical_or((M != 0), (M.T != 0)), np.maximum(M+self.nu, 0), np.maximum(M, 0))
        d_max = np.max(np.sum(M,1)) + 1
        d = np.ones((self.n,)) * d_max
        for i in range(self.n):
            M[i, :] /= d[i]
            M[i, i] = 1. - np.sum(M[i, :])

        # Put 1s back into the diagonal of S_choice tensor (which is zeroed out earlier)
        for i in range(self.n):
            S_choice_tensor[i, i, :] = np.ones((m,))
        return M, d

    def compute_stationary_distribution(self, M, init_pi=None, max_iters=10000, eps=1e-6):
        if init_pi is None:
            pi = np.ones((1,self.n))
        else:
            pi = init_pi
        for _ in range(max_iters):
            pi_ = pi @ M
            if np.linalg.norm(pi_ - pi) < eps:
                break
            pi = pi_
        return pi
