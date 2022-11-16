import numpy as np
import kickscore as ks


class KickScore:
    def __init__(self, n, T):
        self.n = n
        self.T = T
        self.theta_all_times = None
        self.model = None
    
    def fit(self, rankings_all_times, verbose=False, lr=1.0, max_iters=100, kernel_params={}, kernel_generator=None, rank_broken=False):
        print("Performing rank breaking ....")
        
        if kernel_generator is None:
            def kernel_generator(a=1.0, b=0.5, c=1.0):
                return ks.kernel.Constant(var=a) + ks.kernel.Matern52(var=b, lscale=c)
        
        if rank_broken: # Avoid repeated computation
            comparisons_all_times = rankings_all_times
        else:
            comparisons_all_times = self.rank_breaking(rankings_all_times)
            
        model = ks.BinaryModel()
        
        print("Adding items ...")
        for i in range(self.n):
            model.add_item(i, kernel=kernel_generator(**kernel_params))
        
        print("Adding observation ... ")
        for t, comparisons_t in enumerate(comparisons_all_times):
            winners = [i for (i, j) in comparisons_t]
            losers = [j for (i, j) in comparisons_t]
            model.observe(winners=winners, losers=losers, t=t)
            
        print("Fitting model ... ")
        model.fit(verbose=verbose, lr=lr, max_iter=max_iters)
        self.model = model
    
    def get_theta_all_times(self, model):
        theta_all_times = []
        for i in range(self.n):
            theta_i_distr = model.item[i].predict(np.arange(self.T))
            theta_all_times.append(theta_i_distr[0])
        theta_all_times = np.array(theta_all_times).T
        
        theta_all_times -= theta_all_times[:, 0][:, np.newaxis]
        return theta_all_times
    
    def rank_breaking(self, rankings_all_times):
        comparisons_all_times = []
        for t, rankings_t in enumerate(rankings_all_times):
            comparisons_t = []
            for ranking in rankings_t:
                for idi in range(len(ranking)-1):
                    for idj in range(idi+1, len(ranking)):
                        i = ranking[idi]
                        j = ranking[idj]
                        comparisons_t.append((i, j))
            comparisons_all_times.append(comparisons_t)
        return comparisons_all_times