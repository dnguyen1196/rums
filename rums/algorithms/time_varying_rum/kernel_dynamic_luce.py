import numpy as np
from dynamic_synthetic_experiments import T
from rums.algorithms.pl import RegularizedILSR
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax as torch_softmax
import collections
import gpytorch as gp
import torch as th


class KernelDynamicLuce:
    def __init__(self, n, T, utility_kernel, time_kernel):
        self.luce_models = [
            RegularizedILSR(n) for _ in range(T)
        ]
        self.S_all_times = None
        self.rankings_all_times = None
        self.theta_all_times = []
        
    
    def fit(self, rankings_all_times, kernel):
        if self.S_all_times is None or self.rankings_all_times is None:
            self.rankings_all_times = rankings_all_times
            self.construct_choice_tensors_all_times(rankings_all_times)     
        
        # Do kernel time weighting on the rankings given the latent utility estimation
        time_weighting = self.kernel_time_weighting(self.T, kernel)
        
        self.theta_all_times = []
        # Learn the parameters at each time step
        for t in range(self.T):
            weights = []
            weights_wrt_t = time_weighting[t, :] # This should be (T,)
            for tid, S in enumerate(self.S_all_times):
                weights.append(np.ones((len(S),)) * weights_wrt_t[tid])

            lsr = RegularizedILSR(self.n)
            theta_t = lsr.fit(self.S, sample_weights = weights, is_choice_tensor=True)
            self.theta_all_times.append(theta_t - theta_t[0]) # Ensure that item 0 is always at 0
    
    def get_theta_all_times(self):
        return np.array(self.theta_all_times)
    
    def kernel_time_weighting(self, T, kernel):
        time_frame = th.tensor(np.arange(T), dtype=th.float32)
        kTT = kernel(time_frame).detach().numpy()
        kTT /= kTT.sum(1)[:, np.newaxis]
        return kTT
    
    def construct_choice_tensors_all_times(self, rankings_all_times):
        lsr = RegularizedILSR(self.n)
        
        self.S_all_times = []
        for t in range(self.T):
            rankings_t = rankings_all_times[t]
            self.S_all_times.append(lsr.construct_choice_tensor(rankings_t))
            
        self.S = np.concatenate(self.S_all_times, 0) # Concatenate all the S tensors along the 'sample axis'
            
    def time_weighting(self, theta_est_all_times):
        
        
        return
    
    
    