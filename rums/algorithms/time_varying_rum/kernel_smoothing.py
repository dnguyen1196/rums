import numpy as np
from rums.algorithms.pl import RankCentrality


# TODO: one should be able to get the nearest neighbor method my changing the kernel function here.


class KernelASR():
    def __init__(self, n, T):
        self.n = n
        self.T = T
        self.theta_all_times = None

    def fit(self, rankings_all_times, h=1.0):
        pairwise_matrices = self.pairwise_matrices_rank_breaking(rankings_all_times)
        Ps_smoothed = self.kernel_smooth(pairwise_matrices, h)
        self.theta_all_times = self.spectral_ranking(Ps_smoothed)
    
    def pairwise_matrices_rank_breaking(self, rankings_all_times, nu=1):
        matrices = []
        
        for t in range(self.T):
            rankings_t = rankings_all_times[t]
            Pt = np.eye(self.n) * 1./2
            for ranking in rankings_t:
                for idi, i in enumerate(ranking[:-1]):
                    for idj in range(idi+1, len(ranking)):
                        j = ranking[idj]
                        Pt[j, i] += 1

            for i in range(self.n-1):
                for j in range(i+1, self.n):
                    Mij = Pt[i, j] + Pt[j, i]
                    if Mij != 0:
                        Pt[i, j] = (Pt[i, j] + nu)/(Mij + 2 * nu)
                        Pt[j, i] = (Pt[j, i] + nu)/(Mij + 2 * nu)
            matrices.append(Pt)
        return np.array(matrices)
    
    def get_theta_all_times(self):
        return self.theta_all_times - self.theta_all_times[:, 0][:, np.newaxis]
    
    def spectral_ranking(self, Ps):
        theta_all_times = []
        rc = RankCentrality(self.n)
        for P in Ps:
            theta_t = rc.fit_from_pairwise_probabilities(P)
            theta_all_times.append(theta_t)
        return np.array(theta_all_times)
    
    def kernel_function(self, t, tk, h):
        # tk can be a sequence
        return 1/((2 * np.pi)**0.5 * h) * np.exp( - (t - tk)**2 / (2 * h**2))

    def kernel_smooth(self, game_matrix_list, h, T_list = None):
        T, N = game_matrix_list.shape[0:2]
        smoothed = game_matrix_list + 0

        if T_list is None:
            T_list = np.arange(T)

        for t in T_list:
            matrix_this = smoothed[t,:,:]
            tt = (t + 1) / T
            tk = (np.arange(T) + 1) / T
            weight = self.kernel_function(tt,tk,h)
            for i in range(N):
                for j in range(N):
                    matrix_this[i,j] = sum(weight * game_matrix_list[:,i,j])/sum(weight)
            smoothed[t,:,:] = matrix_this
        return smoothed[T_list,:,:]