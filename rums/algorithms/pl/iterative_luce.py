import numpy as np
from scipy.special import softmax
import time

# How should we handle the situation when some items have no outflow -> should we just truncate those at 1./n**2? What's the rationale for this?

class RegularizedILSR():
    def __init__(self, n, lambd=1., nu=1):
        self.n = n
        self.lambd = lambd
        self.theta = None
        self.theta_arr = []
        self.nu = nu # Regularization to avoid the 'rare and bad' item situation
        self.iteration_counts = []

    def fit_old(self, ranked_data, max_iters=100, eps=1e-6, max_iters_mc=10000, eps_mc=1e-6, sample_weights=None, theta_init=None, verbose=False):

        start = time.time()
        if verbose:
            print("Choice breaking ... ")
        choice_data, choice_data_weights = self.choice_break(ranked_data, sample_weights)
        if verbose:
            print(f"Choice breaking took {time.time() - start} seconds")

        if theta_init is None:
            theta_init = np.zeros((self.n,))
        theta = theta_init
        pi = softmax(theta)[:, np.newaxis]
        pi = pi.T
        assert(pi.shape == (1, self.n))

        if verbose:
            print("Running ILSR ... ")

        for it in range(max_iters):
            # Construct Markov chain

            start = time.time()
            M, d = self.construct_markov_chain(choice_data, pi.flatten(), theta, choice_data_weights)
            mc_construction_time = time.time() - start

            # Compute stationary distribution
            start = time.time()
            pi_, iter_counts = self.compute_stationary_distribution(M, max_iters_mc, eps_mc, return_iter_count=True)
            mc_convergence_time = time.time() - start
            self.iteration_counts.append(iter_counts)

            if verbose:
                print(f"ILSR Iter {it}, the MC took {mc_construction_time} to construct and {mc_convergence_time} to converge")

            # Normalize
            # pi_ = pi_/d
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
    
    def construct_markov_chain(self, choice_data, pi, theta, sample_weights=None):
        assert(pi.shape == (self.n,))
        # print(pi)
        if sample_weights is None:
            sample_weights = np.ones((len(choice_data),))

        M = np.zeros((self.n, self.n))

        for sample_idx, (i, choice_sets) in enumerate(choice_data):
            weight = np.sum([pi[k] for k in choice_sets])
            for j in choice_sets:
                if j != i:
                    M[j, i] += 1./weight * sample_weights[sample_idx]

        # Check everypair where if i flows into j, j should also have back flow 
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M+self.nu, M)
        
        if self.lambd > 0:
            d = np.count_nonzero(M, 1)
            for i in range(self.n):
                di = d[i]
                if theta[i] < 0:
                    # Increase flow into state i
                    for j in np.nonzero(M[:, i])[0]:
                        if pi[j] != 0:
                            M[j, i] -= (self.lambd * theta[i]/(di * pi[j]))
                else:
                    # Increase flow from state i
                    for j in np.nonzero(M[i, :])[0]:
                        if pi[i] != 0:
                            M[i, j] += (self.lambd * theta[i]/(di * pi[i]))

        # d = np.count_nonzero(M, 1)
        d_max = np.max(np.sum(M,1))
        d = np.ones((self.n,)) * d_max

        # d = np.sum(M, 1)
        for i in range(self.n):
            M[i, :] /= d[i]
            M[i, i] = 1. - np.sum(M[i, :])

        return M, d


    ###################################################################################################
    #
    #               POTENTIALLY ACCELERATED VERSION
    #
    ###################################################################################################

    def construct_choice_tensor(self, ranked_data):
        # This should return a tensor of size (n, m, n)
        # S where S[j, i, :] = {0,1}^l where S[j, i, l] = 1 if j is ranked ahead of i in pi_l
        S = np.zeros((self.n, self.n, len(ranked_data)))
        for l, rank in enumerate(ranked_data):
            for idj, j in enumerate(rank[:-1]):
                for i in rank[idj:]:
                    S[j, i, l] = 1
        return S

    def construct_markov_chain_accelerated(self, S_choice_tensor, pi, sample_weights=None):
        m = S_choice_tensor.shape[-1]

        if sample_weights is None:
            sample_weights = np.ones((m,))

        M = np.zeros((self.n, self.n))
        # Pre compute pi^T Sk
        piSk = np.zeros((self.n, m))
        for i in range(self.n):
            temp = (pi @ S_choice_tensor[i, :, :]).flatten()
            # piSk[i, :] = np.nan_to_num(1./ temp)
            # Skip division by zero
            piSk[i, :] = np.divide(1, temp, out=np.zeros_like(temp), where=temp!=0)


        # Now we have to zero out all the S_choice_tensor[i, i, :]
        for i in range(self.n):
            S_choice_tensor[i, i, :] = np.zeros((m,))

        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    M[i, j] = np.sum(np.multiply(np.multiply(sample_weights, S_choice_tensor[j, i, :]), piSk[j, :]))

        # Check everypair where if i flows into j, j should also have back flow 
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M+self.nu, M)

        # d = np.count_nonzero(M, 1)
        d_max = np.max(np.sum(M,1))
        d = np.ones((self.n,)) * d_max

        # d = np.sum(M, 1)
        for i in range(self.n):
            M[i, :] /= d[i]
            M[i, i] = 1. - np.sum(M[i, :])

        for i in range(self.n):
            S_choice_tensor[i, i, :] = np.ones((m,))
        return M, d

    def fit_accelerated(self, ranked_data, max_iters=100, eps=1e-6, max_iters_mc=10000, eps_mc=1e-6, sample_weights=None, theta_init=None, verbose=False, is_choice_tensor=False):
        if is_choice_tensor:
            choice_tensor = ranked_data
        else:
            start = time.time()
            choice_tensor = self.construct_choice_tensor(ranked_data)
            if verbose:
                print(f"Constructing the choice tensor took {time.time() - start} seconds")


        if theta_init is None:
            theta_init = np.zeros((self.n,))
        theta = theta_init
        pi = softmax(theta)[:, np.newaxis]
        pi = pi.T
        assert(pi.shape == (1, self.n))

        if verbose:
            print("Running ILSR ... ")

        for it in range(int(max_iters)):
            # Construct Markov chain

            start = time.time()
            M, d = self.construct_markov_chain_accelerated(choice_tensor, pi, sample_weights)
            mc_construction_time = time.time() - start

            # Compute stationary distribution
            start = time.time()
            pi_, iter_counts = self.compute_stationary_distribution(M, pi, max_iters_mc, eps_mc, return_iter_count=True)
            mc_convergence_time = time.time() - start
            self.iteration_counts.append(iter_counts)

            if verbose:
                print(f"ILSR Iter {it}, the MC took {mc_construction_time} to construct and {mc_convergence_time} to converge")

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

    def compute_stationary_distribution(self, M, init_pi=None, max_iters=10000, eps=1e-6, return_iter_count=False):
        if init_pi is None:
            pi = np.ones((1,self.n)) * 1./self.n
        else:
            pi = init_pi
        
        for it in range(max_iters):
            pi_ = pi @ M
            if np.linalg.norm(pi_ - pi) < eps:
                break
            pi = pi_
        if return_iter_count:
            return pi, it
        return pi