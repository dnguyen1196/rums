import numpy as np
from .embeddings import PairwiseEmbedding
from scipy.stats import norm
from scipy.special import softmax, logsumexp
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from sklearn.decomposition import TruncatedSVD
import cvxpy as cp
from scipy.linalg import svd


from sklearn.decomposition import PCA


def heuristic_pair_weight(full_rankings, degree=1, max_w=1.):
    n = len(full_rankings[0])
    embedder = PairwiseEmbedding(n)
    X = embedder.fit(full_rankings)
    
    # Maybe try the first principal component as weights
    # But this is just a direction of maximized variance, then
    # just look at the absolute value of the components
    pca = PCA(n_components=degree)
    pca.fit(X)
    
    Vt = pca.components_ # (degree, d)
    w = pca.explained_variance_ratio_ # (degree,)
    Vt_abs = np.abs(Vt)
    weighted_Vt = np.sum(Vt_abs * w[:, np.newaxis], 0)
    weighted_Vt *= (max_w / np.max(weighted_Vt))
    return weighted_Vt


def m_function_v2(U_not0, X_with_0, X_without_0, X_allpair_not0, q, F, F_prime, lambd=0.00):
    n_minus_1 = len(U_not0)
    m = len(X_with_0)
    Udelta_full = np.outer(U_not0, np.ones((n_minus_1,))) - np.outer(np.ones((n_minus_1,)), U_not0)
    Udelta_unique_pair = Udelta_full[np.triu_indices(n_minus_1, 1)]
    
    # Evaluate function value
    f = 1./m * np.sum(np.log(F(2 * X_without_0 * Udelta_unique_pair)) * q) \
        + 1./m * np.sum(np.log(F(-2 * X_with_0 * U_not0)) * q) \
        - lambd * np.sum(np.square(U_not0)) # Regularization term
    
    # For the gradient, first compute the gradient from the terms involving comparisons among items not 0
    Udelta = Udelta_full.flatten() # Flatten (Row wise)
    
    # should have shape (m, (n-1)**2)
    unweighted_grad = 2 * X_allpair_not0 * F_prime(2 * X_allpair_not0 * Udelta)/ F(2 * X_allpair_not0 * Udelta)
    weighted_grad = np.mean(unweighted_grad * q, 0) # Should have shape (n**2, )
    weighted_grad = np.reshape(weighted_grad, (n_minus_1, n_minus_1)) # Turn into a square (n-1, n-1)
    grad = 0.5 * np.sum(weighted_grad, 1) # We halve it because for every pair i, j, we actually double count the gradient
    
    # For the terms corresponding to comparisons to item 0
    unweighted_grad_with_0 = -2 * X_with_0 * F_prime(-2 * X_with_0 * U_not0)/ F(-2 * X_with_0 * U_not0)
    
    grad += np.mean(unweighted_grad_with_0 * q, 0) # Should have shape (n-1,)
    grad += -2 * lambd * U_not0 # Regularization term
    
    return -f, -grad # Note we're doing minimization, thus the negative signs


class ClusterThenEstimate():
    def __init__(self, n, K, kernel=None, cluster_algo="spectral",
            phi=None, F=None, F_prime=None):
        self.n = n
        self.K = K
        self.embedding = PairwiseEmbedding(n)
        assert(cluster_algo in ["spectral", "sdp"])
        self.cluster_algo = cluster_algo
        self.kernel = kernel
        self.phi = phi
        self.F = F
        self.F_prime = F_prime
        self.opt_solver = "SLSQP"
        
    def fit(self, Pi):
        X = self.embedding.fit(Pi)
        yhat, mus = self.cluster(X)
        # print(X)
        # print(yhat)
        U_hat = []
        mu_hat = []
        self.alpha = []
        
        # Estimate the initial U's by solving an optimization problem
        for k in range(self.K):
            # Get the clustered data
            # Xk = X[np.where(yhat == k)[0],:]
            self.alpha.append(len(X[np.where(yhat == k)[0],:]))
            
            # # Estimate the center
            # muk = np.mean(Xk, 0)
            muk = mus[k, :]
            
            mu_hat.append(muk)
            # print(muk)
            P = np.eye(self.n)
            P[np.triu_indices(self.n, 1)] = muk
            P = -P.T + P
            P += 1./2
            # print(P)

            # Remove extreme values to avoid numerical issues
            P = np.where(P > 0.999, 0.999, P)
            P = np.where(P < 0.001, 0.001, P)

            np.fill_diagonal(P, 0.5)
            Mk = self.phi(P)
            svd = TruncatedSVD(2)
            M_approx = svd.fit_transform(Mk)
            M_approx = M_approx @ svd.components_
            Uk = M_approx[0, :]
            
            # Obtain the initial estimate
            Uk = -(Uk - np.average(Uk)) 
            
            # Then refine furthermore via optimization
            # Uk = self.inference(Xk, Uk)
            Uk = self.inference_from_pairwise_probs(P, Uk)
            
            U_hat.append(Uk)
        
        self.U_all = U_hat
        self.alpha /= np.sum(self.alpha)
        
        return np.array(U_hat)
    
    def spectral_clustering(self, X):
        U, s, Vh = svd(X, full_matrices=False)
        s[self.K:] = 0
        Sigma = np.diag(s)
        Y = (U @ Sigma) @ Vh
        assert(Y.shape == X.shape)
        
        clustering = KMeans(self.K)
        clustering.fit(Y)
        
        return clustering.labels_, clustering.cluster_centers_    
    
    def cluster(self, X):
        return self.spectral_clustering(X)
    
    
    def inference(self, X, U):
        X_allpair = []
        for l in range(len(X)):
            xl_allpair = np.zeros((self.n, self.n))
            xl_allpair[np.triu_indices(self.n, 1)] = X[l, :]
            xl_allpair = (0 - xl_allpair.T) + xl_allpair
            X_allpair.append(xl_allpair.flatten())
        X_allpair = np.array(X_allpair)
        
        m = len(X)
        
        def m_function(Uk):
            Udelta_full = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            Ukdelta = Udelta_full[np.triu_indices(self.n, 1)]
            f = -1./m * np.sum(np.log(self.F(2 * X * Ukdelta))) # Function value (negative Q)
            
            Ukdelta = Udelta_full.flatten() # Flatten (Row wise)
            
            # should have shape (m, n**2)
            unweighted_grad = 2 * X_allpair * self.F_prime(2 * X_allpair * Ukdelta)/ self.F(2 * X_allpair * Ukdelta)
            weighted_grad = unweighted_grad # Should still have shape (m, n**2) 
            weighted_grad = np.sum(weighted_grad, 0) # Should have shape (n**2, )
            
            # Turn into a square (n, n)
            weighted_grad = np.reshape(weighted_grad, (self.n, self.n))
            grad = np.sum(weighted_grad, 1) # Take sum along the column
            
            return f, -1./m * grad # Note we're doing minimization, thus the negative signs
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(m_function, U, method=self.opt_solver, constraints=[constraint], jac=True)
        U = res["x"]
        return U - np.mean(U)
    
    def inference_from_pairwise_probs(self, P, U0):
        # Estimate the parameter using the estimated centers
        np.fill_diagonal(P, 0)
        
        def m_function(Uk):
            Udelta_full = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            f = - np.sum(P * np.log(self.F(Udelta_full)))
            unweighted_grad = P * self.F_prime(Udelta_full)/self.F(Udelta_full)
            grad = np.sum(unweighted_grad, 1) 
            return f, -grad
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(m_function, U0, method=self.opt_solver, constraints=[constraint], jac=True)
        U = res["x"]
        return U - np.mean(U)
    
    
class SpectralSurrogateEM():
    def __init__(self, n, K, kernel=None, init_method="spectral",
            phi=None, F=None, F_prime=None, step_size=0.1, em_method="exact_v1",
            opt_solver="SLSQP", entr_regl=False, 
            tol=1e-3, max_iters=1000, lambd=0.001, use_spectral_centers=True):
        self.n = n
        self.K = K
        
        self.embedding = PairwiseEmbedding(n)
        self.kernel = kernel
        self.init_method = init_method
        assert(init_method in ["sdp", "spectral", "random", "disjoint"])
        self.tol = tol
        self.max_iters = max_iters
        self.step_size = step_size
        self.entr_regl = entr_regl
        self.lambd = lambd # Regularization parameter
        if use_spectral_centers:
            assert(init_method == "spectral")
        self.use_spectral_centers = use_spectral_centers
        self.p_epsilon = 0.001
        
        if phi is None:
            self.phi = lambda d: norm.ppf(d, 0, np.sqrt(2))
        else:
            self.phi = phi
            
        if F is None:
            self.F = lambda x: norm.cdf(x, 0, np.sqrt(2))
        else:
            self.F = F
            
        if F_prime is None:
            self.F_prime = lambda x: norm.pdf(x, 0, np.sqrt(2))
        else:
            self.F_prime = F_prime
            
        if kernel is None:
            # RBF kernel by default
            self.kernel = lambda x, y: np.exp(-0.5 * np.sum((x-y)**2))
        else:
            self.kernel = kernel
            
        assert(em_method in ["exact_v1", "exact_v2", "gradient", "hybrid"])
        self.em_method = em_method
        self.v2 = False
        if "v2" in em_method:
            self.v2 = True
            
        self.opt_solver = opt_solver
            
        self.U_ = None
        self.U_track = []
        self.log_lik_track = []
        self.alpha_track = []
    
    
    def fit(self, Pi, U_init=None, W=None, weighted_pair=True):
        self.weighted_pair = weighted_pair
        X = self.embedding.fit(Pi)
        if weighted_pair:
            self.construct_weights_pair(X, W)
        else:
            self.construct_weights_pos(Pi, W)    
        
        if U_init is None:
            U_init = self.find_initial_solution(X)
        
        self.U_track = []
        self.log_lik_track = []
        self.U = None
        
        if self.em_method == "gradient":
            self.U = self.surrogate_first_order_em(X, U_init)
        elif "exact" in self.em_method:
            self.U = self.surrogate_exact_em(X, U_init)
        else:
            self.U = self.surrogate_hybrid_em(X, U_init)
        
        return self.U, self.alpha_track[-1]
    
    
    def construct_weights_pair(self, X, W=None):
        if W is None:
            self.W = np.ones((len(X[0]),))
        else:
            assert(W.shape == (len(X[0]),))
            self.W = W
            
        self.W_full = np.zeros((self.n, self.n))
        self.W_full[np.triu_indices(self.n, 1)] = self.W
        self.W_full = self.W_full + self.W_full.T
        self.W_full = self.W_full.flatten()
    
    
    def construct_weights_pos(self, Pi, kappa_dict=None):
        d = int(self.n * (self.n - 1)/2)
        self.kappa_weight = []
        self.kappa_weight_full = []      
        for rank in Pi:
            W_arr = np.ones((self.n, self.n))
            for i1, a in enumerate(rank[:-1]):
                for i2 in range(i1+1, len(rank)):
                    b = rank[i2]
                    # if tuple((i1, i2)) in W_dict:
                    W_arr[a, b] = kappa_dict[tuple((i1, i2))]
                    W_arr[b, a] = kappa_dict[tuple((i1, i2))]
            np.fill_diagonal(W_arr, 0)
            self.kappa_weight.append(W_arr[np.triu_indices(self.n, 1)])
            self.kappa_weight_full.append(W_arr.flatten())
        
        self.kappa_weight = np.array(self.kappa_weight)
        self.kappa_weight_full = np.array(self.kappa_weight_full)
            
    ######################################################################################   
    #
    #                                 Initialization
    #
    ######################################################################################     

    
    def spectral_clustering(self, X):
        U, s, Vh = svd(X, full_matrices=False)
        s[self.K:] = 0
        Sigma = np.diag(s)
        Y = (U @ Sigma) @ Vh
        assert(Y.shape == X.shape)
        
        clustering = KMeans(self.K)
        clustering.fit(Y)
        
        return clustering.labels_, clustering.cluster_centers_
    
    def disjoint_pair_clustering(self, X):
        # Partition the set of items into disjoint pairs randomly
        n = self.n if self.n % 2 == 0 else self.n - 1
        disjoint_pairs = np.random.permutation(n).reshape((int(n/2), 2))
        pairs = np.zeros((self.n, self.n))
        for pair in disjoint_pairs:
            i, j = pair[0], pair[1]
            pairs[i, j] = pairs[j, i] = 1
        selected_pairs = pairs[np.triu_indices(self.n, 1)]
        # Then extract the relevant coordinates from X
        X_sub = X[:, np.where(selected_pairs == 1)[0]]
        
        # Then run spectral clustering on the reduced space
        U, s, Vh = svd(X_sub, full_matrices=False)
        s[self.K:] = 0
        Sigma = np.diag(s)
        Y = (U @ Sigma) @ Vh        
        clustering = KMeans(self.K)
        clustering.fit(Y)
        
        return clustering.labels_    
    
    def inference_v1(self, X, U0):
        X_allpair = self.construct_x_allpair(X)
        m = len(X)
        def m_function(Uk):
            Udelta_full = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            Ukdelta = Udelta_full[np.triu_indices(self.n, 1)]
            f = -1./m * np.sum(np.log(self.F(2 * X * Ukdelta)) * self.W) # Function value (negative Q)
            
            Ukdelta = Udelta_full.flatten() # Flatten (Row wise)
            unweighted_grad = 2 * X_allpair * self.F_prime(2 * X_allpair * Ukdelta)/ self.F(2 * X_allpair * Ukdelta) * self.W_full
            weighted_grad = unweighted_grad # Should still have shape (m, n**2) 
            weighted_grad = np.sum(weighted_grad, 0) # Should have shape (n**2, )
            
            # Turn into a square (n, n)
            weighted_grad = np.reshape(weighted_grad, (self.n, self.n))
            grad = np.sum(weighted_grad, 1) # Take sum along the column
            
            return f, -1./m * grad # Note we're doing minimization, thus the negative signs
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(m_function, U0, method=self.opt_solver, constraints=[constraint], jac=True)
        U = res["x"]
        return U - np.mean(U)
    
    
    def inference_v2(self, X, U0):
        m = len(X)
        X_allpair_not0 = self.construct_x_allpair(X)
        assert(X_allpair_not0.shape == (m, (self.n-1)**2))
        X_with_0 = X[:, :self.n-1] # This should have shape (m, n-1)
        X_without_0 = X[:, self.n-1:] # This should have shape (m, nC2 - (n-1))
        
        U_guess = U0[1:] - U0[0] # Note that the variable of this optimization problem only has shape (n-1,)
        # and this is now an unconstrained optimization problem
        res = minimize(m_function_v2, U_guess, method=self.opt_solver, jac=True,
                args=(X_with_0, X_without_0, X_allpair_not0, np.ones((m, 1)), self.F, self.F_prime)
                )        
        U_sub = res["x"]
        U_next = np.concatenate((np.array([0]), U_sub))
        return U_next - np.mean(U_next)
    
    
    def inference_from_pairwise_probs(self, P, U0):
        # Estimate the parameter using the estimated centers
        np.fill_diagonal(P, 0)
        if self.weighted_pair:
            W_full_square = self.W_full.reshape((self.n, self.n))
        else:
            kappa_square = np.reshape(np.mean(self.kappa_weight_full, 0), (self.n, self.n))
        
        def m_function(Uk):
            Udelta_full = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            if self.weighted_pair:
                f = - np.sum(P * np.log(self.F(Udelta_full)) * W_full_square)
                unweighted_grad = P * self.F_prime(Udelta_full)/self.F(Udelta_full) * W_full_square
            else:
                f = - np.sum(P * np.log(self.F(Udelta_full)) * kappa_square)
                unweighted_grad = P * self.F_prime(Udelta_full)/self.F(Udelta_full) * kappa_square
                                
            grad = np.sum(unweighted_grad, 1)
            return f, -grad
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(m_function, U0, method=self.opt_solver, constraints=[constraint], jac=True)
        U = res["x"]
        return U - np.mean(U)
            
    
    def find_initial_solution(self, X):
        mu = None
        if self.init_method == "spectral":
            yhat, mu = self.spectral_clustering(X)
        elif self.init_method == "sdp":
            yhat = self.sdp_clustering(X)
        elif self.init_method == "disjoint":
            yhat = self.disjoint_pair_clustering(X)
        else:
            # Random initialization
            U_hat = np.random.normal(size=(self.K, self.n))
            U_hat = U_hat - np.mean(U_hat, 1)[:, np.newaxis]
            return U_hat
        
        U_hat = []
        
        for k in range(self.K):
            # Get the clustered data
            if not self.use_spectral_centers or (mu is None):
                Xk = X[np.where(yhat == k)[0],:]
                muk = np.mean(Xk, 0)
            else:
                muk = mu[k, :]
                
            # Estimate the center
            P = np.eye(self.n)
            P[np.triu_indices(self.n, 1)] = muk        
            P = -P.T + P
            P += 1./2
            
            # Remove extreme values to avoid numerical issues
            P = np.where(P > 1-self.p_epsilon, 1-self.p_epsilon, P)
            P = np.where(P < self.p_epsilon, self.p_epsilon, P)

            np.fill_diagonal(P, 0.5)
            Mk = self.phi(P)
            svd = TruncatedSVD(2)
            M_approx = svd.fit_transform(Mk)
            M_approx = M_approx @ svd.components_
            Uk = M_approx[0, :]
            
            # Obtain the initial estimate
            Uk = -(Uk - np.average(Uk))
            
            # Then refine more via optimization
            if not self.use_spectral_centers:
                if not self.v2:
                    Uk = self.inference_v1(Xk, Uk)
                else:
                    Uk = self.inference_v2(Xk, Uk)
            else:
                Uk = self.inference_from_pairwise_probs(P, Uk)
            
            U_hat.append(Uk)
        return np.array(U_hat)
    
    ######################################################################################   
    #
    #                           E-step and helper functions
    #
    ###################################################################################### 
        
    def conditional_log_likelihood_pairwise(self, Uk, X):
        Udelta = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
        Udelta = Udelta[np.triu_indices(self.n, 1)]
        if self.weighted_pair:
            log_lik = np.sum(np.log(self.F(2 * X * Udelta)) * self.W, 1)
        else:
            log_lik = np.sum(np.log(self.F(2 * X * Udelta)) * self.kappa_weight, 1)
        return log_lik
    
    
    def surrogate_e_step(self, X, U, alpha):
        # Estimate qz
        joint_log_lik = np.zeros((len(X), self.K))
        for k in range(self.K):
            log_lik = self.conditional_log_likelihood_pairwise(U[k, :], X)
            joint_log_lik[:, k] = log_lik + np.log(alpha[k]) # np.log(1./self.K)
        qz = softmax(joint_log_lik, 1)
        # Compute loglikelihood
        # log_lik = np.sum(qz * joint_log_lik)
        log_lik = np.mean(logsumexp(joint_log_lik, 1))
        return qz, log_lik
    
    
    def surrogate_e_step_entropic_regularized(self, X, U, alpha):
        # Following http://www.cs.cmu.edu/~tom/10-702/Zoubin-702.pdf
        # Estimate the log joint likelihood
        m = len(X)
        d = len(X[0]) # (n choose 2)
        joint_log_lik = np.zeros((m, self.K))
        for k in range(self.K):
            log_lik = self.conditional_log_likelihood_pairwise(U[k, :], X)
            joint_log_lik[:, k] = log_lik + np.log(alpha[k])
        
        qz = cp.Variable((m,self.K))
        obj = cp.sum(cp.multiply(qz, joint_log_lik)) + cp.sum(cp.entr(qz))
        constraints = [
            qz @ np.ones((self.K,)) == np.ones((m,)), # Normalization constraint,
            qz <= 1.,
            qz >= 0.
        ]
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(verbose=False)
        qz = qz.value
        assert(qz.shape == (m, self.K))
        log_lik = np.sum(qz * joint_log_lik)
        return qz, log_lik
    
    
    def construct_x_allpair(self, X):
        X_allpair = []
        if not self.v2:
            for l in range(len(X)):
                xl_allpair = np.zeros((self.n, self.n))
                xl_allpair[np.triu_indices(self.n, 1)] = X[l, :]
                xl_allpair = (0 - xl_allpair.T) + xl_allpair
                X_allpair.append(xl_allpair.flatten())
            X_allpair = np.array(X_allpair)
        else:
            for l in range(len(X)):
                xl_allpair = np.zeros((self.n-1, self.n-1))
                # Only take the comparisons starting from the second item
                xl_allpair[np.triu_indices(self.n-1, 1)] = X[l, self.n-1:]
                xl_allpair = (0 - xl_allpair.T) + xl_allpair
                X_allpair.append(xl_allpair.flatten())
            X_allpair = np.array(X_allpair)
        return X_allpair
    
    ######################################################################################   
    #
    #                                       Exact M-step (This is the best performing)
    #
    ######################################################################################    
    
    def surrogate_exact_em(self, X, U_init):
        U = U_init
        alpha = np.ones((self.K,)) / self.K
        self.U_track = [U]
        self.alpha_track = [alpha]
        
        X_allpair = self.construct_x_allpair(X)
        X_with_0 = X[:, :self.n-1] if self.v2 else None
        X_without_0 = X[:, self.n-1:] if self.v2 else None
        
        for _ in range(self.max_iters):
            U_next, log_lik, alpha_next = self.surrogate_em_step(X, X_allpair, U, alpha, X_with_0, X_without_0)
            U = (1 - self.step_size) * U + self.step_size * U_next
            alpha = (1 - self.step_size) * alpha + self.step_size * alpha_next
            
            self.log_lik_track.append(log_lik)
            self.U_track.append(U)
            self.alpha_track.append(alpha)
            
            if len(self.U_track) > 0 and np.linalg.norm(U - self.U_track[-1], "fro") < self.tol:
                break
            
        self.U = U
        self.alpha = alpha
        return U
        
    def surrogate_em_step(self, X, X_allpair, U, alpha, X_with_0=None, X_without_0=None):
        # Estimate qz
        if self.entr_regl:
            qz, log_lik = self.surrogate_e_step_entropic_regularized(X, U, alpha)
        else:
            qz, log_lik = self.surrogate_e_step(X, U, alpha)
        alpha = np.mean(qz, 0)
        alpha /= np.sum(alpha)
        
        # Do M-step
        U_next = []
        for k in range(self.K):
            if not self.v2: # Version 1 of the M-step
                Uk_next = self.exact_m_step_v1(U[k, :], X, X_allpair, qz[:, k], X_with_0, X_without_0)
            else: # Version 2
                Uk_next = self.exact_m_step_v2(U[k, :], X, X_allpair, qz[:, k], X_with_0, X_without_0)
            U_next.append(Uk_next)
            
        return np.array(U_next), log_lik, alpha

    def exact_m_step_v1(self, Uk, X, X_allpair, qzk, X_with_0=None, X_without_0=None):
        m = len(X)
        
        def m_function(Uk):
            Ukdelta_full = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            Ukdelta = Ukdelta_full[np.triu_indices(self.n, 1)]
            Ukdelta_full_flatten = Ukdelta_full.flatten() # Flatten (Row wise)

            if self.weighted_pair:
                f = -1./m * np.sum(np.log(self.F(2 * X * Ukdelta)) * qzk[:, np.newaxis] * self.W) - self.lambd * np.sum(np.square(Uk)) # Function value (negative Q)
                unweighted_grad =  2 * X_allpair * self.F_prime(2 * X_allpair * Ukdelta_full_flatten)/ self.F(2 * X_allpair * Ukdelta_full_flatten) * self.W_full
            else:
                f = -1./m * np.sum(np.log(self.F(2 * X * Ukdelta)) * qzk[:, np.newaxis] * self.kappa_weight) - self.lambd * np.sum(np.square(Uk))
                unweighted_grad =  2 * X_allpair * self.F_prime(2 * X_allpair * Ukdelta_full_flatten)/ self.F(2 * X_allpair * Ukdelta_full_flatten) * self.kappa_weight_full

            weighted_grad = unweighted_grad * qzk[:, np.newaxis] # Should still have shape (m, n**2)
            weighted_grad = np.sum(weighted_grad, 0) # Should have shape (n**2, )
            
            # Turn into a square (n, n)
            weighted_grad = np.reshape(weighted_grad, (self.n, self.n))
            # Zero out the diagonal because we also have F_prime(0) which may not be zero
            np.fill_diagonal(weighted_grad, 0)
            
            grad = -1./m * np.sum(weighted_grad, 1) # Take row sums
            grad += self.lambd * 0.5 * Uk # Gradient from the regularization term
            return f, grad # Note we're doing minimization, thus the negative signs
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)        
        
        res = minimize(m_function, Uk, method=self.opt_solver, constraints=[constraint], jac=True)
        Uk_next = res["x"]
        return Uk_next - np.mean(Uk_next) # Return the normalized partworths
    
    def exact_m_step_v2(self, Uk, X, X_allpair_not0, qzk, X_with_0, X_without_0):
        """
        Version 2: we only have n-1 parameters as we implicitly fix U_0 = 0
        
        Need to make sure that X_allpair_not0 is shape (m, (n-1)**2)
        
        """
        m = len(X)
        assert(X_allpair_not0.shape == (m, (self.n-1)**2))
        
        U0 = Uk[1:] - Uk[0] # Note that the variable of this optimization problem only has shape (n-1,)
        # and this is now an unconstrained optimization problem
        res = minimize(m_function_v2, U0, method=self.opt_solver, jac=True,
                       args=(X_with_0, X_without_0, X_allpair_not0, qzk[:, np.newaxis], self.F, self.F_prime),
                       )
        Uk_sub_next = res["x"]
        Uk_next = np.concatenate((np.array([0]), Uk_sub_next))
        return Uk_next - np.mean(Uk_next)
            
    ######################################################################################   
    #
    #                                       First order M-step
    #
    ######################################################################################
    
    def surrogate_first_order_em(self, X, U_init):
        U = U_init
        alpha = np.ones((self.K,)) / self.K
        X_allpair = self.construct_x_allpair(X)
        
        for _ in range(self.max_iters):
            gradU, log_lik = self.first_order_em_step(X, X_allpair, U, alpha)
            U = U + self.step_size * gradU # Consider dynamic step size
            
            # Normalize the partworths to sum to 0
            U = U - np.mean(U, 1)[:, np.newaxis]
            
            if len(self.U_track) > 0 and np.linalg.norm(U - self.U_track[-1], "fro") < self.tol:
                self.log_lik_track.append(log_lik)
                self.U_track.append(U)
                break

            self.log_lik_track.append(log_lik)
            self.U_track.append(U)
        
        return U
        
    def gradient_q_func(self, X_allpair, Uk, qzk):
        # Compute the gradient with respect to U
        m = len(X_allpair)
        
        Ukdelta = np.outer(Uk, np.ones((self.n, ))) - np.outer(np.ones((self.n,)), Uk)
        Ukdelta = Ukdelta.flatten() # Flatten (Row wise)
        
        # should have shape (m, n**2)
        unweighted_grad = 2 * X_allpair * self.F_prime(2 * X_allpair * Ukdelta)/ self.F(2 * X_allpair * Ukdelta)
        weighted_grad = unweighted_grad * qzk[:, np.newaxis] # Should still have shape (m, n**2) 
        weighted_grad = np.sum(weighted_grad, 0) # Should have shape (n**2, )
        
        # Turn into a square (n, n)
        weighted_grad = np.reshape(weighted_grad, (self.n, self.n))
        grad = np.sum(weighted_grad, 1) # Take sum along the column
        return 1./m * grad
    
    def first_order_em_step(self, X, X_allpair, U, alpha):
        qz, log_lik = self.surrogate_e_step(X, U, alpha)
        
        # Compute gradient
        U_grad = []
        for k in range(self.K):
            partial_Uk = self.gradient_q_func(X_allpair, U[k, :], qz[:, k])
            U_grad.append(partial_Uk)
            
        return np.array(U_grad), log_lik
    
    ######################################################################################   
    #
    #                                   Surrogate EM
    #
    ######################################################################################
     
    def surrogate_hybrid_em(self, X, U_init):
        # First do the exact surrogate EM to obtain a good estimate
        U = self.surrogate_exact_em(X, U_init)
        # Then run first order EM
        return self.surrogate_first_order_em(X, U)
    
    
class SpectralPairwiseSurrogateEM(SpectralSurrogateEM):
    def __init__(self, n, K, kernel=None, init_method="spectral",
            phi=None, F=None, F_prime=None, step_size=1.,
            opt_solver="SLSQP", lambd=1.,
            tol=1e-3, max_iters=1000, use_spectral_centers=True, include_priors=True):
        super(SpectralPairwiseSurrogateEM, self).__init__(n, K, kernel=kernel, init_method=init_method, em_method="exact_v1",
            phi=phi, F=F, F_prime=F_prime, step_size=step_size, opt_solver=opt_solver, tol=tol, max_iters=max_iters, 
            use_spectral_centers=use_spectral_centers, lambd=lambd)
        self.include_priors = include_priors
    
    
    def surrogate_em_step(self, X, X_allpair, U, alpha, X_with_0=None, X_without_0=None):
        qz, log_lik = self.surrogate_e_step(X, U, alpha)
        alpha = np.ones((self.K,)) * 1./self.K # TODO: fix this
        
        U_next = []
        for k in range(self.K):
            Uk_next = self.exact_m_step(X, qz[:, :, k], U[k, :], X_allpair)
            U_next.append(Uk_next)
        
        return np.array(U_next), log_lik, alpha
    
    
    def exact_m_step(self, X, qzk, Uk, X_allpair):
        m, d = qzk.shape
        n = len(Uk)
        qzk_allpair = np.zeros((m, self.n, self.n))
        
        for l in range(m):
            qzk_allpair[l][np.triu_indices(self.n, 1)] = qzk[l, :]
        
        qzk_allpair = qzk_allpair + np.transpose(qzk_allpair, (0,2,1)) # This should have shape (m, n, n) still
        qzk_allpair = np.reshape(qzk_allpair, (m, self.n**2))
        
        def q_function(Uk):
            Ukdelta_full = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            Ukdelta = Ukdelta_full[np.triu_indices(self.n, 1)]
            phi = np.log(self.F(2 * X * Ukdelta)) * self.W # This should have shape (m, d)
            # qzk have shape (m, d)
            E_qzk_phi = qzk * phi # This should have shape (m, d)
            f = -np.mean(np.sum(E_qzk_phi, 1)) + self.lambd * np.sum(np.square(Uk)) # Function evaluation
            
            # Now compute the derivative
            Ukdelta_full = Ukdelta_full.flatten() 
            unweighted_grad = 2 * X_allpair  * self.F_prime(2 * X_allpair * Ukdelta_full) / self.F(2 * X_allpair * Ukdelta_full) * self.W_full # This has shape (m, n*2)
            weighted_grad = qzk_allpair * unweighted_grad
            weighted_grad = np.reshape(weighted_grad, (m, self.n, self.n))
            
            for l in range(m):
                np.fill_diagonal(weighted_grad[l], 0)
            
            # np.fill_diagonal(weighted_grad, 0)
            grad = np.sum(weighted_grad, 2) # This should have shape (m, n)
            grad = -np.mean(grad, 0) # This should have shape (n,)
            grad += 0.5 * self.lambd * Uk # Normalization term
            return f, grad
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)        
        res = minimize(q_function, Uk, method=self.opt_solver, constraints=[constraint], jac=True)
        Uk_next = res["x"]
        return Uk_next - np.mean(Uk_next) # Return the normalized partworths
            
        
    def surrogate_e_step(self, X, U, alpha):
        m, d = X.shape
        K, n = U.shape
        
        q_approximate = np.zeros((m, d, K))
        
        for k in range(self.K):
            Uk = U[k, :]
            Udelta = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk)
            Udelta = Udelta[np.triu_indices(self.n, 1)]
            # Compute log-likelihood for each pair P(Yij, z|Theta)
            if self.weighted_pair:
                joint_log_lik = np.log(self.F(2 * X * Udelta)) * self.W # This should have shape (m, d)
            else:
                joint_log_lik = np.log(self.F(2 * X * Udelta)) * self.kappa_weight # This should have shape (m,d)
            
            if self.include_priors:
                joint_log_lik = joint_log_lik + np.log(alpha[k])
            q_approximate[:, :, k] = joint_log_lik
        
        # Take soft-max of the log likelihood 
        q_approximate = softmax(q_approximate, 2) 
        assert(q_approximate.shape == (m, d, K))
        
        log_lik_yij = np.sum(q_approximate, 2) # This should have shape (m, d)
        log_lik = np.mean(np.sum(np.log(log_lik_yij), 1)) # This should have shape ()
        return q_approximate, log_lik