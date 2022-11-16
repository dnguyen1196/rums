import argparse

parser = argparse.ArgumentParser("Synthetic dynamic PL experiments")
parser.add_argument("--out_folder", type=str, default="./experiments_results/time_varying", help="Output folder")
parser.add_argument("--n", type=int, default=20, help="Number of items")
parser.add_argument("--T", type=int, default=30, help="Number of time steps")


parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--seed", type=int, default=119, help="Random seed")
parser.add_argument("--n_fix", type=int, default=5, help="Number of items with fixed utility")
parser.add_argument("--pn", type=float, default=1., help="p * n for sampling")

parser.add_argument("--our", action="store_true", default=False, help="Our method")
parser.add_argument("--smoothing", action="store_true", default=False, help="Kernel smoothing")
parser.add_argument("--kickscore", action="store_true", default=False, help="Kick score")

args = parser.parse_args()

from rums.data import random_utility_models as rum
from rums.data import synthetic as synthetic_rum
from rums.algorithms.time_varying_rum import DynamicLuce, ThetaParam, RegularizedWeaver, KernelASR, GPBackEnd, KickScore
import rums
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")
import torch as th
from scipy.special import softmax
from torch.nn.functional import softmax as torch_softmax
import os
import kickscore as ks


out_folder = args.out_folder
n = args.n
T = args.T
n_trials = args.n_trials
seed = args.seed
n_fix = args.n_fix
pn = args.pn

our = args.our
smoothing = args.smoothing
kickscore = args.kickscore

np.random.seed(seed)

theta_all_times = []
theta = np.random.normal(size=(n,))
theta_all_times = np.array([theta for t in range(T)])
# Replace some of the items by sinuisoid functions
for i in range(n_fix, n):
    theta_all_times[:, i] += np.random.normal(0, 1) * np.sin(np.arange(T)/2)

theta_all_times -= theta_all_times[:, 0][:, np.newaxis]
p_all_times = softmax(theta_all_times, 1)


p_pair = pn/n
menus_all_times = []
for t in range(T):
    menus_all_times.append(synthetic_rum.generate_pairs_erdos_renyi(n, p_pair))

##################################################################################################################3

def kl_ptilde_p_torch(p, p_tilde):
    # KL(ptilde || p)
    return th.nn.functional.kl_div(p_tilde, p, size_average=False)

def kl_ptilde_p_derivative_torch(p, p_tilde):
    # Derivative with respect to p
    return p_tilde/p

##################################################################################################################

our_results = {}
smoothing_results = {}

m_array = [25, 50, 100, 200] #, 300, 400, 500, 1000, 2000]


def save_partial_results():
    methods = ""
    
    if our:
        methods += "_our"
    if smoothing:
        methods += "_smoothing"
    if kickscore:
        methods += "_kickcore"

    output_file = os.path.join(out_folder, f"syn_n={n}_sd={seed}_mds={methods}.pkl")

    # Save results
    th.save({
        "menus_all_times" : menus_all_times,
        "p_pair" : p_pair,
        "true_theta": theta_all_times,
        "our" : our_results,
        "smoothing" : smoothing_results,
        "m_array" : m_array,
    }, output_file)


n_inducing_choices = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
kernel = "Matern"
mean_function = "Constant"
our_results = {
    "n_inducing_choices" : n_inducing_choices,
    "kernel" : "Matern",
    "mean_function" : "Constant",
    "errors_by_m": [],
    "time_by_m" : []
}
    
h_choices = [0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
smoothing_results = {
    "h_choices" : h_choices, 
    "errors_by_m" : [],
    "time_by_m" : []
}

kickscore_a_choices =  [0.01, 0.05, 0.1, 0.5, 1.0]
kickscore_b_choices = [0.1, 0.25, 0.5, 1.0, 1.5]
kickscore_c_choices = [0.1, 0.5, 1, 2, 5]

kickscore_results = {
    "kickscore_b_choices" : kickscore_b_choices,
    "kickscore_a_choices" : kickscore_a_choices,
    "kickscore_c_choices" : kickscore_c_choices,
    "errors_by_m" : [],
    "time_by_m" : []
}


def kickscore_kernel_generator(a=1.0, b=0.5, c=1.0):
    return ks.kernel.Constant(var=a) + ks.kernel.Matern52(var=b, lscale=c)

for m_per_time in m_array:
    rankings_all_times = synthetic_rum.generate_timed_partial_rankings_from_partworths(theta_all_times, menus_all_times, rum.gumbel_std, m_per_time=m_per_time)


    
    if our:
        F_GP = GPBackEnd(n, T, 10, kernel=kernel, mean_function=mean_function)
        dynamic_model = DynamicLuce(n, T, F_GP, kl_ptilde_p_torch, kl_ptilde_p_derivative_torch, rho=1, lambd=0.1)
        # Pre compute the a, b, Delta and connected components
        a_all_times, b_all_times, Delta_all_times, connected_components_all_times = dynamic_model.construct_a_b_Delta_connected_components_all_times(rankings_all_times)
        rankings_all_times_dict = {
            "a_all_times" : a_all_times,
            "b_all_times" : b_all_times,
            "Delta_all_times" : Delta_all_times,
            "connected_components_all_times": connected_components_all_times
        }
        
        errors_by_n_inducing = []
        time_track = []
        for n_inducing in n_inducing_choices:
            try:
                F_GP = GPBackEnd(n, T, n_inducing, kernel=kernel, mean_function=mean_function)
                dynamic_model = DynamicLuce(n, T, F_GP, kl_ptilde_p_torch, kl_ptilde_p_derivative_torch, rho=1, lambd=0.1)
                start = time.time()
                dynamic_model.fit(rankings_all_times_dict, max_iters=100, verbose=False, eps=1e-4, loss_eps=1, 
                                    report=1, step_size=1, lr=0.01, lr_decay=1.0, weight_decay=0.001, max_theta_iters=1000, reference_theta=theta_all_times.flatten())
                time_track.append(time.time() - start)
                
                est_theta = dynamic_model.get_theta_all_times()
                errors_by_n_inducing.append(np.sum(np.square(est_theta - theta_all_times)))
            except Exception as e:
                errors_by_n_inducing.append(np.nan)
                
        our_results["errors_by_m"].append(errors_by_n_inducing)
        our_results["time_by_m"].append(time_track)
        
    if smoothing:
        kernel_asr = KernelASR(n, T)
        errors_by_h = []
        time_track = []
        for h in h_choices:
            start = time.time()
            kernel_asr.fit(rankings_all_times, h=h)
            time_track.append(time.time() - start)
            est_theta = kernel_asr.get_theta_all_times()
            errors_by_h.append(np.sum(np.square(est_theta - theta_all_times)))
            
        smoothing_results["errors_by_m"].append(errors_by_h)
        smoothing_results["time_by_m"].append(time_track)
        
    if kickscore:
        kickscore = KickScore(n, T)
        comparisons_all_times = kickscore.rank_breaking(rankings_all_times) # Do all the rank breaking in advance to avoid wasting computations
        
        errors_track = []
        time_track = []
        
        for a in kickscore_a_choices:
            for b in kickscore_b_choices:
                for c in kickscore_c_choices:
                    kernel_params = {
                        "a":a,
                        "b":b,
                        "c":c,
                    }
                    start = time.time()
                    kickscore.fit(comparisons_all_times, verbose=True, lr=.95, max_iters=100, kernel_params=kernel_params, kernel_generator=kickscore_kernel_generator, rank_broken=True)
                    time_track.append(time.time() - start)
                    est_theta = kickscore.get_theta_all_times(kickscore.model)
                    errors_track.append(np.sum(np.square(est_theta - theta_all_times)))
                    
        kickscore_results["time_by_m"].append(time_track)
        kickscore_results["errors_by_m"].append(errors_track)

    save_partial_results()