import argparse

parser = argparse.ArgumentParser("Mixtures of PL experiments on real datasets")
parser.add_argument("--out_folder", type=str, default="./experiment_results/", help="Output folder")
parser.add_argument("--dataset", type=str, default="sushi", help="Dataset name", 
    choices=[
            "sushi", "irish_meath", "irish_north", "irish_west", "f1", "apa",
            "ml_100k", "ml_1m", "ml_10m", "ml_20m", "book_genome"
    ])
parser.add_argument("--dataset_extra", type=int, default=1, help="Extra identifier for datasets (usually the year number)")
parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data to the whole datasets")
parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation data to training data")
parser.add_argument("--seed", type=int, default=119, help="Random seed")
parser.add_argument("--max_iters", type=int, default=100, help="Max iterations of EM")
parser.add_argument("--max_num_items", type=int, default=500, help="Max number of items")

parser.add_argument("--lsr_1", action="store_true", default=False, help="LSR(1) with RC")
parser.add_argument("--lsr_2", action="store_true", default=False, help="LSR(2) with GMM")
parser.add_argument("--lsr_3", action="store_true", default=False, help="LSR(1) with Cluster")
parser.add_argument("--lsr_4", action="store_true", default=False, help="LSR(1) with Cluster and trimmed loss")
parser.add_argument("--lsr_init", type=str, default="cluster", help="Init type", choices=["rc", "gmm", "cluster", "lrmc"])

parser.add_argument("--cml", action="store_true", default=False, help="CML")
parser.add_argument("--gmm", action="store_true", default=False, help="GMM")
parser.add_argument("--emm", action="store_true", default=False, help="EMM")
parser.add_argument("--bayesian", action="store_true", default=False, help="Bayesian")
parser.add_argument("--cluster", action="store_true", default=False, help="Cluster only")
parser.add_argument("--random", action="store_true", default=False, help="Random guess")

parser.add_argument("--small", action="store_true", default=False, help="Small sample sizes")


args = parser.parse_args()
out_folder = args.out_folder
dataset = args.dataset
dataset_extra = args.dataset_extra
seed = args.seed
train_ratio = args.train_ratio
max_iters = args.max_iters
val_ratio = args.val_ratio
max_num_items = args.max_num_items
small = args.small
lsr_init = args.lsr_init

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
from rums.evaluation import log_likelihood
from rums.data import random_utility_models as rum
import rums
import rums.data.data_loader as data
import rums.data.large_data_loader as large_data
import torch as th
import os
import numpy as np
import time

phi = lambda x: rums.algorithms.utils.phi_gumbel(x, 1)
F = lambda x: rums.algorithms.utils.F_gumbel(x, 1)
F_prime = lambda x: rums.algorithms.utils.F_prime_gumbel(x, 1)


np.random.seed(seed)

if "irish" in dataset:
    election = dataset.split("_")[1]
    ordering, all_rankings = getattr(data, "irish")(election, seed=seed)
elif "apa" in dataset:
    ordering, all_rankings = getattr(data, "apa")(year=dataset_extra, seed=seed)
elif dataset in ["ml_100k", "ml_1m", "ml_10m", "ml_20m", "book_genome"]:
    ordering, all_rankings = getattr(large_data, dataset)(num_movies=max_num_items, seed=None)
else:
    ordering, all_rankings = getattr(data, dataset)(seed=seed)

n = len(ordering)
m_total = len(all_rankings)

m_training = int(m_total * train_ratio)
m_testing = m_total - m_training
m_validation = int(m_training * val_ratio)
m_inference = m_training - m_validation

if small:
    m_ratio_array = np.array([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3])
else:
    m_ratio_array = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

m_array =  [int(m_r * m_inference) for m_r in m_ratio_array]
K_array = [2,3,4,5,6,8,10]

# For each trial seed, partition the data into training and testing set, and within training set
# Partion into inference and validation set
perm = np.random.permutation(m_total)
train_indices = perm[:m_training]
test_indices = perm[m_training:]

all_train_rankings = [all_rankings[i] for i in train_indices]
test_rankings = [all_rankings[i] for i in test_indices]
validation_rankings = all_train_rankings[m_inference:]

lambd_1, nu_1 = 0., 0.001
# lambd_2, nu_2 = 1., 0.
# lambd_3, nu_3 = 0.01, 0.01
# lambd_4, nu_4 = 0.01, 1.

val_loglik_bym_cml = []
val_loglik_bym_lsr_1 = []
val_loglik_bym_lsr_2 = []
val_loglik_bym_lsr_3 = []
val_loglik_bym_lsr_4 = []
val_loglik_bym_gmm = []
val_loglik_bym_emm = []
val_loglik_bym_bayesian = []
val_loglik_bym_cluster = []
val_loglik_bym_random = []


test_loglik_bym_cml = []
test_loglik_bym_lsr_1 = []
test_loglik_bym_lsr_2 = []
test_loglik_bym_lsr_3 = []
test_loglik_bym_lsr_4 = []
test_loglik_bym_gmm = []
test_loglik_bym_emm = []
test_loglik_bym_bayesian = []
test_loglik_bym_cluster = []
test_loglik_bym_random = []


time_bym_em_cml = []
time_bym_em_lsr_1 = []
time_bym_em_lsr_2 = []
time_bym_em_lsr_3 = []
time_bym_em_lsr_4 = []
time_bym_em_gmm = []
time_bym_emm = []
time_bym_bayesian = []


its_bym_cml = []
its_bym_lsr_1 = []
its_bym_lsr_2 = []
its_bym_lsr_3 = []
its_bym_lsr_4 = []
its_bym_gmm = []
its_bym_emm = []



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

    output_file = os.path.join(out_folder, 
            f"ds={dataset}+{dataset_extra}_sd={seed}_m={m_total}_n={n}_mds={methods}_its={max_iters}.th")

    # Save results
    th.save({
        "m_total" : m_total,
        "K_array" : K_array,
        "m_array" : m_array,

        "val_lsr" : val_loglik_bym_lsr_1,
        "val_lsr_2" : val_loglik_bym_lsr_2,
        "val_lsr_3" : val_loglik_bym_lsr_3,
        "val_lsr_4" : val_loglik_bym_lsr_4,
        "val_cml" : val_loglik_bym_cml,
        "val_gmm" : val_loglik_bym_gmm,
        "val_emm" : val_loglik_bym_emm,
        "val_bayesian" : val_loglik_bym_bayesian,
        "val_cluster" : val_loglik_bym_cluster,
        "val_random" : val_loglik_bym_random,
        

        "test_lsr" : test_loglik_bym_lsr_1,
        "test_lsr_2" : test_loglik_bym_lsr_2,
        "test_lsr_3" : test_loglik_bym_lsr_3,
        "test_lsr_4" : test_loglik_bym_lsr_4,
        "test_cml" : test_loglik_bym_cml,
        "test_gmm" : test_loglik_bym_gmm,
        "test_emm" : test_loglik_bym_emm,
        "test_bayesian" : test_loglik_bym_bayesian,
        "test_cluster" : test_loglik_bym_cluster,
        "test_random" : test_loglik_bym_random,
        
        
        "param" : (lambd_1, nu_1),
        # "param_2" : (lambd_2, nu_2),
        # "param_3" : (lambd_3, nu_3),
        # "param_4" : (lambd_4, nu_4),

        "time_cml" : time_bym_em_cml,
        "time_gmm" : time_bym_em_gmm,
        "time_bayesian" : time_bym_bayesian,

        "time_lsr" : time_bym_em_lsr_1,
        "time_lsr_2" : time_bym_em_lsr_2,
        "time_lsr_3" : time_bym_em_lsr_3,
        "time_lsr_4" : time_bym_em_lsr_4,
        "time_emm" : time_bym_emm,
        
        "its_cml" : its_bym_cml,
        "its_gmm" : its_bym_gmm,
        "its_lsr" : its_bym_lsr_1,
        "its_lsr_2" : its_bym_lsr_2,
        "its_lsr_3" : its_bym_lsr_3,
        "its_lsr_4" : its_bym_lsr_4,
        "its_emm" : its_bym_emm,

    }, output_file)

U_random_init_all = []

for K in K_array:
    U_random_init = np.random.normal(0, 1, size=(K, n))
    U_random_init = U_random_init - np.mean(U_random_init, 1)[:, np.newaxis]
    U_random_init_all.append(U_random_init)

for idm, m in enumerate(m_array):
    inference_rankings = all_train_rankings[:m]

    val_loglik_em_cml = []
    val_loglik_em_lsr_1 = []
    val_loglik_em_lsr_2 = []
    val_loglik_em_lsr_3 = []
    val_loglik_em_lsr_4 = []
    val_loglik_em_gmm = []
    val_loglik_emm = []
    val_loglik_bayesian = []
    val_loglik_cluster = []
    val_loglik_random = []


    test_loglik_em_cml = []
    test_loglik_em_lsr_1 = []
    test_loglik_em_lsr_2 = []
    test_loglik_em_lsr_3 = []
    test_loglik_em_lsr_4 = []
    test_loglik_em_gmm = []
    test_loglik_emm = []
    test_loglik_bayesian = []
    test_loglik_cluster = []
    test_loglik_random = []


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

    for idk, K in enumerate(K_array):
        # EM-CML
        if cml:
            em_cml = EM_CML(n, K)
            start = time.time()
            U_em_cml, alpha_cml = em_cml.fit(inference_rankings, max_iters=max_iters)
            time_em_cml += [time.time() - start]
            its_em_cml += [len(em_cml.U_array)]
            val_loglik_em_cml += [log_likelihood(validation_rankings, U_em_cml, alpha_cml)]
            test_loglik_em_cml += [log_likelihood(test_rankings, U_em_cml, alpha_cml)]

        # EM-LSR cluster
        
        if lsr_1:
            em_lsr_1 = SpectralEM(n, K, lambd_1, nu_1, init_method=lsr_init, extra_refinement=True)
            start = time.time()
            U_em_lsr_1, alpha_lsr_1 = em_lsr_1.fit(inference_rankings, max_iters=max_iters)
            time_em_lsr_1 += [time.time() - start]
            its_em_lsr_1 += [em_lsr_1.settings()]
            val_loglik_em_lsr_1 += [log_likelihood(validation_rankings, U_em_lsr_1, alpha_lsr_1)]
            test_loglik_em_lsr_1 += [log_likelihood(test_rankings, U_em_lsr_1, alpha_lsr_1)]

        if lsr_2:
            em_lsr_2 = SpectralEM(n, K, lambd_1, nu_1, init_method=lsr_init, extra_refinement=True, hard_e_step=True)
            start = time.time()
            U_em_lsr_2, alpha_lsr_2 = em_lsr_2.fit(inference_rankings, max_iters=max_iters)
            time_em_lsr_2 += [time.time() - start]
            its_em_lsr_2 +=  [em_lsr_2.settings()]
            val_loglik_em_lsr_2 += [ log_likelihood(validation_rankings, U_em_lsr_2, alpha_lsr_2)]
            test_loglik_em_lsr_2 += [log_likelihood(test_rankings, U_em_lsr_2, alpha_lsr_2)]

        if lsr_3:
            em_lsr_3 = SpectralEM(n, K, lambd_1, nu_1, init_method=lsr_init, trimmed_llh=True, trimmed_threshold=np.exp(-n), extra_refinement=True)
            start = time.time()
            U_em_lsr_3, alpha_lsr_3 = em_lsr_3.fit(inference_rankings, max_iters=max_iters)
            time_em_lsr_3 += [time.time() - start]
            its_em_lsr_3 +=  [em_lsr_3.settings()]
            val_loglik_em_lsr_3 += [ log_likelihood(validation_rankings, U_em_lsr_3, alpha_lsr_3)]
            test_loglik_em_lsr_3 += [log_likelihood(test_rankings, U_em_lsr_3, alpha_lsr_3)]

        if lsr_4:
            em_lsr_4 = SpectralEM(n, K, lambd_1, nu_1, init_method=lsr_init, trimmed_llh=True, trimmed_threshold=np.exp(-n), extra_refinement=False, hard_e_step=True)
            start = time.time()
            U_em_lsr_4, alpha_lsr_4 = em_lsr_4.fit(inference_rankings, max_iters=max_iters)
            time_em_lsr_4 += [time.time() - start]
            its_em_lsr_4 +=  [em_lsr_4.settings()]
            val_loglik_em_lsr_4 += [ log_likelihood(validation_rankings, U_em_lsr_4, alpha_lsr_4)]
            test_loglik_em_lsr_4 += [log_likelihood(test_rankings, U_em_lsr_4, alpha_lsr_4)]
        
        # EM-GMM
        if gmm:
            em_gmm = EM_GMM_PL(n, K, step_size=1.)
            start = time.time()
            U_em_gmm, alpha_gmm = em_gmm.fit(inference_rankings, max_iters=max_iters)
            time_em_gmm += [time.time() - start]
            its_em_gmm += [len(em_gmm.U_array)]
            val_loglik_em_gmm += [log_likelihood(validation_rankings, U_em_gmm, alpha_gmm)]
            test_loglik_em_gmm += [log_likelihood(test_rankings, U_em_gmm, alpha_gmm)]

        # EMM
        if emm:
            em_lsr = SpectralEM(n, K, lambd=0, nu=0.001)
            U_init = em_lsr.spectral_init(inference_rankings)
            emm = EMM(n, K)
            start = time.time()
            U_emm, alpha_emm = emm.fit(inference_rankings, U_init=U_init, max_iters=max_iters)
            time_emm += [time.time() - start]
            its_emm += [len(emm.U_array)]
            val_loglik_emm += [log_likelihood(validation_rankings, U_emm, alpha_emm)]
            test_loglik_emm += [log_likelihood(test_rankings, U_emm, alpha_emm)]
            

        # EMM
        if bayesian:
            em_lsr = SpectralEM(n, K, lambd=0, nu=0.001)
            U_init = em_lsr.spectral_init(inference_rankings)
            bayesian_est = BayesianEMGS(n, K)
            start = time.time()
            U_bayesian, alpha_bayesian = bayesian_est.fit_then_sample_estimate(inference_rankings, U_init, n_samples=100)
            time_bayesian += [time.time() - start]                
            val_loglik_bayesian += [log_likelihood(validation_rankings, U_bayesian, alpha_bayesian)]
            test_loglik_bayesian += [log_likelihood(test_rankings, U_bayesian, alpha_bayesian)]
            
            
        if cluster:
            em_lsr = SpectralEM(n, K, lambd=0, nu=0.001)
            U_cluster = em_lsr.spectral_init(inference_rankings)
            alpha = np.ones((K,)) * 1./K
            qz = em_lsr.e_step(inference_rankings, U_cluster, alpha)
            alpha_cluster = np.mean(qz, 0)
            alpha_cluster /= np.sum(alpha_cluster)
            val_loglik_cluster += [log_likelihood(validation_rankings, U_cluster, alpha_cluster)]
            test_loglik_cluster += [log_likelihood(test_rankings, U_cluster, alpha_cluster)]
            
        if random:
            U_random_init = U_random_init_all[idk]
            U_random = U_random_init - np.mean(U_random_init, 1)[:, np.newaxis]
            alpha_random = np.ones((K,)) * 1./K
            val_loglik_cluster += [log_likelihood(validation_rankings, U_random, alpha_random)]
            test_loglik_cluster += [log_likelihood(test_rankings, U_random, alpha_random)]
            

    val_loglik_bym_cml += [val_loglik_em_cml]
    val_loglik_bym_lsr_1 += [val_loglik_em_lsr_1]
    val_loglik_bym_lsr_2 += [val_loglik_em_lsr_2]
    val_loglik_bym_lsr_3 += [val_loglik_em_lsr_3]
    val_loglik_bym_lsr_4 += [val_loglik_em_lsr_4]
    val_loglik_bym_gmm += [val_loglik_em_gmm]
    val_loglik_bym_emm += [val_loglik_emm]
    val_loglik_bym_bayesian += [val_loglik_bayesian]
    val_loglik_bym_cluster += [val_loglik_cluster]
    val_loglik_bym_random += [val_loglik_random]


    test_loglik_bym_cml += [test_loglik_em_cml]
    test_loglik_bym_lsr_1 += [test_loglik_em_lsr_1]
    test_loglik_bym_lsr_2 += [test_loglik_em_lsr_2]
    test_loglik_bym_lsr_3 += [test_loglik_em_lsr_3]
    test_loglik_bym_lsr_4 += [test_loglik_em_lsr_4]
    test_loglik_bym_gmm += [test_loglik_em_gmm]
    test_loglik_bym_emm += [test_loglik_emm]
    test_loglik_bym_bayesian += [test_loglik_bayesian]
    test_loglik_bym_cluster += [test_loglik_cluster]
    test_loglik_bym_random += [test_loglik_random]


    time_bym_em_cml += [time_em_cml]
    time_bym_em_gmm += [time_em_gmm]
    time_bym_em_lsr_1 += [time_em_lsr_1]
    time_bym_em_lsr_2 += [time_em_lsr_2]
    time_bym_em_lsr_3 += [time_em_lsr_3]
    time_bym_em_lsr_4 += [time_em_lsr_4]
    time_bym_bayesian += [time_bayesian]
    time_bym_emm += [time_emm]
    
    its_bym_cml += [its_em_cml]
    its_bym_lsr_1 += [its_em_lsr_1]
    its_bym_lsr_2 += [its_em_lsr_2]
    its_bym_lsr_3 += [its_em_lsr_3]
    its_bym_lsr_4 += [its_em_lsr_4]
    its_bym_gmm += [its_em_gmm]
    its_bym_emm += [its_emm]
    
    # Save partial results
    save_partial_results()
