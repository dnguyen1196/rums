import numpy as np
import cvxpy as cp
import scipy as sp
from scipy.special import softmax
from scipy.optimize import minimize, LinearConstraint

class TimeVaryingPL():
    def __init__(self, n, T, H, lambd=1, nu=1):
        self.n = n
        self.T = T
        self.H = H
        self.lambd = lambd
        self.nu = nu
        self.betas_array = []

    def fit(self, rankings_by_time, rho=0.1, alpha=1., beta_init=None, max_iters=100, eps=1e-6):
        assert(len(rankings_by_time) == self.T)
        if beta_init is None:
            beta = np.zeros((self.n, self.T))
        else:
            beta = beta_init
        y = np.zeros((self.n, self.T))
        pi = np.ones((self.n, self.T)) * 1./n
        assert(beta.shape == (self.n, self.T))

        # Assume that rankings_by_time is a list of T lists,
        # t-th sublist is the rankings collected at time step t
        for it in range(max_iters):
            pi_next = []
            for t in range(self.T):
                pi_t = ilsr_minimize_pi(rankings_by_time[t], beta[:, t], y[:, t], rho=rho)
                pi_next.append(pi_t)
            
            pi_next = np.array(pi_next)
            beta_next = self.minimize_beta(pi_next, y, beta, self.H, rho=rho, alpha=alpha)
            y_next = self.update_dual_time_step(y, pi_next, beta_next, rho)

            if np.sum((beta_next - beta)**2) < eps:
                break

            pi = pi_next
            beta = beta_next
            y = y_next
            self.betas_array.append(beta)

        return beta

    def minimize_beta(self, pi, y, cur_beta, H, rho=0.1, alpha=1.):
        """ minimize rho * ||pi - softmax(beta)||_2^2 + alpha * H(beta) + y^T (pi - softmax(beta))

        NOTE That this function optimizes over all time steps
        """
        assert(pi.shape == cur_beta.shape)
        assert(y.shape == cur_beta.shape)
        assert(cur_beta.shape == (self.n * self.T,))

        def loss(beta):
            obj = alpha * H(beta)
            obj += np.dot(y, pi - softmax(beta))
            obj += rho * np.sum((pi - softmax(beta))**2)
            return obj

        # Impose constraint beta_0^{(0)} = 0
        onehot = np.zeros_like(cur_beta)[:, np.newaxis]
        onehot[0, 0] = 1
        constraint = LinearConstraint(onehot.T, 0, 0)
        res = minimize(loss, cur_beta, constraints=[constraint])
        beta = res['x']
        return beta

    def update_dual_time_step(self, y, pi, beta, rho=0.1):
        return y + rho * (pi - softmax(beta, 0))

    def ilsr_minimize_pi(self, ranked_data, beta_t, y_t, rho=0.1, max_iters=100, eps=1e-6, max_iters_mc=10000, eps_mc=1e-6, sample_weights=None, theta_init=None):
        """ Fit one time step
        """
        choice_data, choice_data_weights = self.choice_break(ranked_data, sample_weights)
        if theta_init is None:
            theta_init = np.zeros((self.n,))
        theta = theta_init
        pi = softmax(theta)[:, np.newaxis]
        pi = pi.T
        assert(pi.shape == (1, self.n))

        for _ in range(max_iters):
            # Construct Markov chain
            M, d = self.construct_markov_chain_regularized(choice_data, pi.flatten(), beta_t, y_t, rho, choice_data_weights)
            # Compute stationary distribution
            pi_ = self.compute_stationary_distribution(M, max_iters_mc, eps_mc)

            # Normalize
            pi_ = pi_/d
            pi_[0, :] = pi_[0, :] / np.sum(pi_)

            # Estimate item parameters
            theta_ = np.log(pi_.flatten())
            theta_ -= np.mean(theta_)

            if np.linalg.norm(pi.flatten() - pi_.flatten()) < eps:
                break

            pi = pi_
            theta = theta_
            self.theta_arr.append(theta)

        self.theta = np.copy(theta)
        return theta
    
    def choice_break(self, data, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones((len(data),))

        choice_data = []
        choice_data_weights = []
        for idpi, rank in enumerate(data):
            for idx, i in enumerate(rank[:-1]):
                choice_data.append((i, rank[idx:]))
                choice_data_weights.append(sample_weights[idpi])

        return choice_data, np.array(choice_data_weights)

    def construct_markov_chain(self, choice_data, pi, beta, y, rho, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones((len(choice_data),))
        beta_bar = softmax(beta)

        M = np.zeros((self.n, self.n))

        for sample_idx, (i, choice_sets) in enumerate(choice_data):
            weight = np.sum([pi[k] for k in choice_sets])
            for j in choice_sets:
                if j != i:
                    M[j, i] += 1./weight * sample_weights[sample_idx]

        # Check everypair where if i flows into j, j should also have back flow 
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M+self.nu, M)

        if self.lambd > 0:
            d = np.maximum(np.count_nonzero(M, 1))
            for i in range(self.n):
                di = d[i]
                yi = y[i]
                if yi > 0:
                    for j in np.nonzero(M[i, :])[0]:
                        M[i, j] += yi/di + rho * pi[i]/di
                    for j in np.nonzero(M[:, i])[0]:
                        M[j, i] += rho * beta_bar[i] /(pi[j] * di) if pi[j] > 0 else 0
                else:
                    for j in np.nonzero(M[i, :])[0]:
                        M[i, j] += rho * pi[i]/di
                    for j in np.nonzero(M[:, i])[0]:
                        M[j, i] += -yi/di * pi[i]/pi[j] + rho * beta_bar[i]/(pi[j] * di) if pi[j] > 0 else 0

        d_max = np.max(np.sum(M,1))
        d = np.ones((self.n,)) * d_max

        # d = np.sum(M, 1)
        for i in range(self.n):
            M[i, :] /= d[i]
            M[i, i] = 1. - np.sum(M[i, :])

        return M, d

    def compute_stationary_distribution(self, M, max_iters=10000, eps=1e-6):
        pi = np.ones((1,self.n))
        for _ in range(max_iters):
            pi_ = pi @ M
            if np.linalg.norm(pi_ - pi) < eps:
                break
            pi = pi_
        return pi

    def find_connected_components(self, menus_by_time):
        connected_components_by_time = []

        def find_representative(representative, i):
            if representative[i] == i:
                return i
            else:
                return find_representative(representative, representative[i])

        def find_connected_components_(menus, n):
            representative = dict([(i, i) for i in range(n)])

            for menu in menus: # Check every pair in every menu
                for i in menu:
                    for j in menu:
                        if i < j:
                            repi = find_representative(representative, i)
                            repj = find_representative(representative, j)

                            # If they belong to different components, then merge
                            if repj != repi:
                                representative[j] = repi
            # Now tally all the nodes with the same representatives
            connected_components = {}
            for i in range(n):
                repi = find_representative(representative, i)
                if repi not in connected_components:
                    connected_components[repi] = [i]
                else:
                    connected_components[repi].append(i)

            connected_components = [
                components for key, components in connected_components.items() if len(components) > 1
            ]

            return connected_components

        for t in range(self.T):
            menus_at_time_t = menus_by_time[t]
            connected_components_by_time.append(find_connected_components_(menus_at_time_t))

        return connected_components_by_time