import numpy as np


def generate_partial_ranked_data(U, menus, noise, m=200, seed=None):
    rankings = []
    for menu in menus:
        menu_size = len(menu)
        new_idx_to_old = dict([(i, item) for i, item in enumerate(menu)])
        U_sub = np.array([U[i] for i in menu])
        
        X = np.zeros((m, menu_size))
        for i in range(menu_size):
            X[:, i] = np.repeat(U_sub[i], m) + noise((m,))

        ranked = np.argsort(X, 1)
        # Highest item first
        ranked = ranked[:, ::-1]        
        for ranking in ranked:
            rankings.append(
                [new_idx_to_old[i] for i in ranking]
            )

    return rankings


def generate_random_menus(n, menu_sizes):
    all_menus = set()
    for i in range(len(menu_sizes)):
        menu = tuple(set(np.random.choice(n, size=(menu_sizes[i],), replace=False)))
        if menu in all_menus:
            i -= 1
        else:
            all_menus.add(menu)

    return all_menus


def generate_rum_ranked_data(U, noises, n_samples=1000, p0=1.):
    n = len(U)
    X = np.zeros((n_samples, n))
    
    for i in range(n):
        X[:, i] = np.repeat(U[i], n_samples) + noises[i]((n_samples,))

    ranked = np.argsort(X, 1)
    # Highest item first
    ranked = ranked[:, ::-1]
    
    # Missing data
    if p0 != 1.:
        sub_ranked = []
        # Generate a mask
        mask = np.random.rand(n_samples, n)
        mask[np.where(mask > 1-p0)] = 1
        mask[np.where(mask < p0)] = 0
        for i, rank in enumerate(ranked):
            partial_ranking = rank[np.where(mask[i] == 1)]
            if len(partial_ranking) <= 1: # Don't include vacuous rankings
                continue
            sub_ranked.append(partial_ranking)
        ranked = sub_ranked
    else:
        ranked = [rank for rank in ranked]
    return ranked


def generate_iid_rum_ranked_data(U, noise, n_samples=1000, p0=1.):
    noises = [noise for _ in range(len(U))]
    return generate_rum_ranked_data(U, noises, n_samples, p0)


def generate_mixture_rum_ranked_data(U_all, alpha, noise, m=200, p0=1., mixture_id=False, seed=None):
    np.random.seed(seed)
    K = len(alpha)
    ranked_data = []
    n = len(U_all[0])
    num_per_mixture = np.random.multinomial(m, alpha)
    which_mixture = []
    for k, m_k in enumerate(num_per_mixture):
        mixture_ranked_data = generate_iid_rum_ranked_data(U_all[k], noise, m_k, p0)
        ranked_data += mixture_ranked_data
        for i in range(m_k):
            which_mixture.append(k)

    if mixture_id:
        return ranked_data, np.array(which_mixture)
    
    return [x for x in ranked_data]


def generate_pairs_erdos_renyi(n, p):
    pairs = []
    for i in range(n-1):
        for j in range(i+1, n):
            if np.random.rand() < p:
                pairs.append((i, j))

    return pairs


def generate_time_varying_utilities(n, T, fre_sigma=1, bias_sigma=1, seed=None):
    U_all = []
    np.random.seed(seed)
    # Generate sine waves
    frequency = np.random.rand(n) * fre_sigma
    bias = np.random.rand(n) * bias_sigma
    for i in range(n):
        Ui = np.sin(np.arange(T) * frequency[i]) + bias[i]
        U_all.append(Ui)
    return np.array(U_all)


def generate_time_varying_rum_comparisons(U_all, noise, L=100, p0=None, seed=None):
    n, T = U_all.shape
    data_by_time = []
    if p0 is None:
        p0 = 1./n # Then with high probability the comparison graph is not connected

    for t in range(T):
        data_at_time_t = []
        # Generate Erdos-Renyi graph with prob p0
        pairs = generate_pairs_erdos_renyi(n, p0)
        print(pairs)
        data_at_time_t.extend(
            generate_partial_ranked_data(U_all[:, t], pairs, noise, L)
        )
        data_by_time.append(data_at_time_t)

    return data_by_time

