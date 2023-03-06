import argparse

parser = argparse.ArgumentParser("Synthetic mixtures of PL experiments")
parser.add_argument("--out_folder", type=str, default="./experiment_results/", help="Output folder")
parser.add_argument("--n", type=int, default=100, help="Number of items")
parser.add_argument("--K", type=int, default=2, help="Number of mixtures")

parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
parser.add_argument("--sigma", type=float, default=2., help="Spread of the Us")
parser.add_argument("--seed", type=int, default=119, help="Random seed")
parser.add_argument("--model",type=str, default="random", choices=["random", "semi-random", "semi-det", "top-items"])
parser.add_argument("--num_informative", type=int, default=10, help="Number of informative items")
parser.add_argument("--max_iters", type=int, default=100, help="Max iterations")

parser.add_argument("--lsr_1", action="store_true", default=False, help="LSR(1) with RC")
parser.add_argument("--lsr_2", action="store_true", default=False, help="LSR(2) with GMM")
parser.add_argument("--lsr_3", action="store_true", default=False, help="LSR(1) with Cluster")
parser.add_argument("--lsr_4", action="store_true", default=False, help="LSR(1) with Cluster")

parser.add_argument("--cml", action="store_true", default=False, help="CML")
parser.add_argument("--gmm", action="store_true", default=False, help="GMM")
parser.add_argument("--emm", action="store_true", default=False, help="EMM")
parser.add_argument("--bayesian", action="store_true", default=False, help="Bayesian")
parser.add_argument("--cluster", action="store_true", default=False, help="Cluster only")
parser.add_argument("--random", action="store_true", default=False, help="Random guessing")


args = parser.parse_args()
out_folder = args.out_folder
n = args.n
n_trials = args.n_trials
sigma = args.sigma
seed = args.seed
model = args.model
max_iters = args.max_iters
lsr_1 = args.lsr_1
lsr_2 = args.lsr_2
lsr_3 = args.lsr_3
lsr_4 = args.lsr_4

cml = args.cml
gmm = args.gmm
emm = args.emm
bayesian = args.bayesian
cluster = args.cluster
random = args.random


from rums.algorithms.mixtures import SpectralEM, EM_GMM_PL, EMM, EM_CML, BayesianEMGS
from rums.data.synthetic import generate_mixture_rum_ranked_data
from rums.evaluation import ell2_mixture, clustering_accuracy
from rums.data import random_utility_models as rum
import rums
import numpy as np
import matplotlib.pyplot as plt
import time
import torch as th
import os


p0 = 1
K = args.K
alpha = np.array([1./K for _ in range(K)])
N = 1

sigma = 1
scale_prior = 2

phi = lambda x: rums.algorithms.utils.phi_gumbel(x, 1)
F = lambda x: rums.algorithms.utils.F_gumbel(x, 1)
F_prime = lambda x: rums.algorithms.utils.F_prime_gumbel(x, 1)
noise = rum.gumbel_std

np.random.seed(seed)


####################################################################
#
#                   GENERATE MODEL PARAMETERS
#
####################################################################

num_informative = args.num_informative
num_uninformative = n - num_informative

# Generate model parameters
U_true = []
mu_true = []
for k in range(K):
    if model == "random": # Completely random model
        U = np.random.normal(0, scale_prior, (n,))
    elif model == "semi-random": # Semi-random model
        U = np.concatenate(
                (
                    np.zeros((num_uninformative,)),
                    np.random.normal(0, scale_prior, (num_informative,))
                )
            )
    elif model == "top-items": # Only differing by the top items
        U_top = np.random.normal(0, scale_prior, (num_informative,))
        U_top -= np.min(U_top)
        U = np.concatenate(
                (
                    np.zeros((num_uninformative,)),
                    U_top
                )
            )
    else: # Divided model
        which_item = np.random.choice(n, (num_informative,), replace=False)
        U = np.zeros((n,))
        for i in which_item:
            U[i] = scale_prior
    mu = rum.compute_mean_pairwise(U, F)
    U_true.append(U)
    mu_true.append(mu)
    
U_true = np.array(U_true)
U_true = U_true - np.mean(U_true, 1)[:, np.newaxis]
mu_true = np.array(mu_true)

min_delta = np.inf
for i in range(K-1):
    for j in range(i+1, K):
        delta = np.linalg.norm(U_true[i,:] - U_true[j, :])
        if delta < min_delta:
            min_delta = delta

print("Delta = ", min_delta )

m_array = [100,200,300,400,500, 1000, 2000]

lambd_1, nu_1 = 0, 0.001
lambd_2, nu_2 = 1., 0.
lambd_3, nu_3 = 0.01, 0.01
lambd_4, nu_4 = 0.01, 1.

err_bym_cml = []
err_bym_lsr_1 = []
err_bym_lsr_2 = []
err_bym_lsr_3 = []
err_bym_lsr_4 = []
err_bym_gmm = []
err_bym_emm = []
err_bym_bayesian = []
err_bym_cluster = []
err_bym_random = []


time_bym_cml = []
time_bym_lsr_1 = []
time_bym_lsr_2 = []
time_bym_lsr_3 = []
time_bym_lsr_4 = []
time_bym_gmm = []
time_bym_emm = []
time_bym_bayesian = []


its_bym_cml = []
its_bym_lsr_1 = []
its_bym_lsr_2 = []
its_bym_lsr_3 = []
its_bym_lsr_4 = []
its_bym_gmm = []
its_bym_emm = []
its_bym_bayesian = []


def save_partial_results():
    methods = ""
    if lsr_1:
        methods += "_lsr_1"
    if lsr_2:
        methods += "_lsr_2"
    if lsr_3:
        methods += "_lsr_3"
    if lsr_4:
        methods += "_lsr_4"
    if cml:
        methods += "_cml"
    if gmm:
        methods += "_gmm"
    if emm:
        methods += "_emm"
    if bayesian:
        methods += "_bayesian"
    if cluster:
        methods += "_cluster"
    if random:
        methods += "_random"

    output_file = os.path.join(out_folder, f"syn_n={n}_K={K}_sd={seed}_mdl={model}_num={num_informative}_mds={methods}_its={max_iters}.th")

    # Save results
    th.save({
        "m_array" : m_array,
        "U" : U,
        "cml" : err_bym_cml,
        "gmm" : err_bym_gmm,
        "lsr_1" : err_bym_lsr_1,
        "lsr_2" : err_bym_lsr_2,
        "lsr_3" : err_bym_lsr_3,
        "lsr_4" : err_bym_lsr_4,
        "emm" : err_bym_emm,
        "bayesian": err_bym_bayesian,
        "cluster": err_bym_cluster,
        "random": err_bym_random,

        "time_cml" : time_bym_cml,
        "time_gmm" : time_bym_gmm,
        "time_lsr_1" : time_bym_lsr_1,
        "time_lsr_2" : time_bym_lsr_2,
        "time_lsr_3" : time_bym_lsr_3,
        "time_lsr_4" : time_bym_lsr_4,
        "time_emm" : time_bym_emm,
        "time_bayesian": time_bym_bayesian,
        
        "its_cml" : its_bym_cml,
        "its_gmm" : its_bym_gmm,
        "its_lsr_1" : its_bym_lsr_1,
        "its_lsr_2" : its_bym_lsr_2,
        "its_lsr_3" : its_bym_lsr_3,
        "its_lsr_4" : its_bym_lsr_4,
        "its_emm" : its_bym_emm,

        "param_1" : (lambd_1, nu_1),
        "param_2" : (lambd_2, nu_2),
        "param_3" : (lambd_3, nu_3),
        "param_4" : (lambd_4, nu_4),

    }, output_file)


U_random_init = np.random.normal(0, 1, size=(K, n))
U_random_init = U_random_init - np.mean(U_random_init, 1)[:, np.newaxis]

np.random.seed(seed)

for m in m_array:
    err_em_cml = []
    err_em_lsr_1 = []
    err_em_lsr_2 = []
    err_em_lsr_3 = []
    err_em_lsr_4 = []
    err_em_gmm = []
    err_emm = []
    err_bayesian = []
    err_cluster = []
    err_random = []


    time_em_cml = []
    time_em_lsr_1 = []
    time_em_lsr_2 = []
    time_em_lsr_3 = []
    time_em_lsr_4 = []
    time_em_gmm = []
    time_emm = []
    time_bayesian = []

    
    its_em_cml = []
    its_em_lsr_1 = []
    its_em_lsr_2 = []
    its_em_lsr_3 = []
    its_em_lsr_4 = []
    its_em_gmm = []
    its_emm = []


    for trial in range(n_trials):
        rankings, y_true = generate_mixture_rum_ranked_data(U_true, alpha, noise, m, mixture_id=True)

        # EM-CML
        if cml:
            em_cml = EM_CML(n, K)
            start = time.time()
            U_em_cml, _ = em_cml.fit(rankings, max_iters=max_iters)
            time_em_cml += [time.time() - start]
            its_em_cml += [len(em_cml.U_array)]
            err_em_cml += [np.linalg.norm(ell2_mixture(U_true, U_em_cml))]
        
        # EM-LSR
        if lsr_1:
            em_lsr_1 = SpectralEM(n, K, lambd_1, nu_1, init_method="cluster")
            start = time.time()
            U_em_lsr_1, _  = em_lsr_1.fit(rankings, max_iters=max_iters)
            time_em_lsr_1 += [time.time() - start]
            its_em_lsr_1 +=  [em_lsr_1.settings()]
            err_em_lsr_1 += [np.linalg.norm(ell2_mixture(U_true, U_em_lsr_1))]

        if lsr_2:
            em_lsr_2 = SpectralEM(n, K, lambd_1, nu_1, init_method="cluster", hard_e_step=True)
            start = time.time()
            U_em_lsr_2, _  = em_lsr_2.fit(rankings, max_iters=max_iters)
            time_em_lsr_2 += [time.time() - start]
            its_em_lsr_2 +=  [em_lsr_2.settings()]
            err_em_lsr_2 += [np.linalg.norm(ell2_mixture(U_true, U_em_lsr_2))]

        if lsr_3:
            em_lsr_3 = SpectralEM(n, K, lambd_1, nu_1, init_method="cluster", trimmed_llh=True, trimmed_threshold=np.exp(-(n)))
            start = time.time()
            U_em_lsr_3, _  = em_lsr_3.fit(rankings, max_iters=max_iters)
            time_em_lsr_3 += [time.time() - start]
            its_em_lsr_3 += [em_lsr_3.settings()]
            err_em_lsr_3 += [np.linalg.norm(ell2_mixture(U_true, U_em_lsr_3))]
        
        if lsr_4:
            em_lsr_4 = SpectralEM(n, K, lambd_1, nu_1, init_method="cluster", trimmed_llh=True, trimmed_threshold=np.exp(-(n)), hard_e_step=True)
            start = time.time()
            U_em_lsr_4, _  = em_lsr_4.fit(rankings, max_iters=max_iters)
            time_em_lsr_4 += [time.time() - start]
            its_em_lsr_4 += [em_lsr_4.settings()]
            err_em_lsr_4 += [np.linalg.norm(ell2_mixture(U_true, U_em_lsr_4))]
            
        # EM-GMM
        if gmm:
            em_gmm = EM_GMM_PL(n, K)
            start = time.time()
            U_em_gmm, _  = em_gmm.fit(rankings, max_iters=max_iters)
            time_em_gmm += [time.time() - start]
            its_em_gmm += [len(em_gmm.U_array)]
            err_em_gmm += [np.linalg.norm(ell2_mixture(U_true, U_em_gmm))]

        # EMM
        if emm:
            emm = EMM(n, K)
            start = time.time()
            U_emm, alpha_emm = emm.fit(rankings, U_init=U_random_init, max_iters=max_iters)
            time_emm += [time.time() - start]
            its_emm += [len(emm.U_array)]
            err_emm += [np.linalg.norm(ell2_mixture(U_true, U_emm))]
            
        # Bayesian
        if bayesian:
            em_lsr = SpectralEM(n, K, lambd=0, nu=0.001)
            U_init = em_lsr.spectral_init(rankings)
            bayesian_est = BayesianEMGS(n, K)
            start = time.time()
            try:
                U_bayesian, alpha_bayesian = bayesian_est.fit_then_sample_estimate(rankings, U_init, n_samples=100)
                time_bayesian += [time.time() - start]
                err_bayesian += [np.linalg.norm(ell2_mixture(U_true, U_bayesian))]
            except Exception as e:
                pass
        
        if cluster:
            em_lsr = SpectralEM(n, K, lambd=0, nu=0.001)
            U_cluster = em_lsr.spectral_init(rankings)
            err_cluster += [np.linalg.norm(ell2_mixture(U_true, U_cluster))]
        
        if random:
            U_random_init = np.random.normal(0, 1, size=(K, n))
            U_random_init = U_random_init - np.mean(U_random_init, 1)[:, np.newaxis]
            err_random += [ell2_mixture(U_true, U_random_init)]
            

    err_bym_cml += [err_em_cml]
    err_bym_lsr_1 += [err_em_lsr_1]
    err_bym_lsr_2 += [err_em_lsr_2]
    err_bym_lsr_3 += [err_em_lsr_3]
    err_bym_lsr_4 += [err_em_lsr_4]
    err_bym_gmm += [err_em_gmm]
    err_bym_emm += [err_emm]
    err_bym_bayesian += [err_bayesian]
    err_bym_cluster += [err_cluster]
    err_bym_random += [err_random]


    time_bym_cml += [time_em_cml]
    time_bym_lsr_1 += [time_em_lsr_1]
    time_bym_lsr_2 += [time_em_lsr_2]
    time_bym_lsr_3 += [time_em_lsr_3]
    time_bym_lsr_4 += [time_em_lsr_4]
    time_bym_gmm += [time_em_gmm]
    time_bym_emm += [time_emm]
    time_bym_bayesian += [time_bayesian]

    
    its_bym_cml += [its_em_cml]
    its_bym_lsr_1 += [its_em_lsr_1]
    its_bym_lsr_2 += [its_em_lsr_2]
    its_bym_lsr_3 += [its_em_lsr_3]
    its_bym_lsr_4 += [its_em_lsr_4]
    its_bym_gmm += [its_em_gmm]
    its_bym_emm += [its_emm]
    
    save_partial_results()