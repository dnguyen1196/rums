import numpy as np
import scipy as sp
import os
import collections
import nimfa
import ast
import torch as th


def factorize(V, rank=30):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The MovieLens data matrix. 
    :type V: `numpy.matrix`
    """
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=rank, max_iter=30, version='r', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    return fit.basis(), fit.coef()


def lrmc(A, rank=30):
    # Assume that A is (num_users, num_items)
    n, m = A.shape
    W, H = factorize(A)
    A_estimate = W @ H
    return A_estimate


def construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users=None, num_movies=None):
    user_id_list = [(user_id, ratings_count_user[user_id]) for user_id in user_id_set]
    movie_id_list = [(movie_id, ratings_count_movie[movie_id]) for movie_id in movie_id_set]
    user_id_list = sorted(user_id_list, key=lambda x: x[1])[::-1]
    movie_id_list = sorted(movie_id_list, key=lambda x: x[1])[::-1]

    if num_users is None:
        num_users = len(user_id_list)
    if num_movies is None: 
        num_movies = len(movie_id_list)

    user_id_list = user_id_list[:min(num_users, len(user_id_list))]
    movie_id_list = movie_id_list[:min(num_movies, len(movie_id_list))]

    user_mapping = dict([(user_id, i) for i, (user_id, _) in enumerate(user_id_list)])
    movie_mapping = dict([(movie_id, i) for i, (movie_id, _) in enumerate(movie_id_list)])
    
    m, n = len(user_mapping), len(movie_mapping)
    A = np.zeros((m, n))

    for user, movie, rating in ratings:
        if user in user_mapping and movie in movie_mapping:
            uid = user_mapping[user]
            mid = movie_mapping[movie]
            A[uid, mid] = rating

    # Remove row with all zeros
    all_zeros_row = np.all(A == 0, 1)
    non_zeros_row = np.logical_not(all_zeros_row)
    A = A[non_zeros_row, :]

    A = sp.sparse.csr_matrix(A)
    return A


def rankings_from_completed_matrix(A_lr):
    # Then produce an estimate ranking over all items
    A_lr_dense = A_lr.todense()
    all_rankings = []
    m, n = A_lr.shape

    for i in range(m):
        pi = list(np.argsort(np.ravel( A_lr_dense[i, :])))[::-1]
        all_rankings.append(pi)
    return np.arange(n), all_rankings



def extract_sub_rankings(all_rankings, n_items, selected_size, seed=None):
    sub_rankings = []
    
    if seed is not None:
        print("Random seed given, selecting items at random")
        np.random.seed(seed)
        selected_items = np.random.choice(n_items, size=(selected_size), replace=False)
    else:
        print("Random seed not specified, picking items with highest variance in rankings ... ")
        # Select the items with the highest variance in rankings
        rankings_by_items = collections.defaultdict(list)
        for ranking in all_rankings:
            for r, item in enumerate(ranking):
                rankings_by_items[item].append(r)
        ranking_variances = [np.var(rankings_by_items[item]) for item in rankings_by_items.keys()]
        selected_items = np.argsort(ranking_variances)[-selected_size:]
        
    idx_change = dict([(old_idx, new_idx) for new_idx, old_idx in enumerate(selected_items)])
    
    for rank in all_rankings:
        sub_rankings.append([idx_change[item] for item in rank if item in selected_items])
    return sub_rankings


def ml_100k(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None):
    path = "datasets/ratings_datasets/ml-100k-ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    f.readline()
    ratings = []
    movie_id_set = set()
    user_id_set = set()

    ratings_count_user = collections.defaultdict(int)
    ratings_count_movie = collections.defaultdict(int)

    for line in f:
        data = line.rstrip().split(',') # Note the seperator
        user_id = int(data[0])
        movie_id = int(data[1])
        rating = float(data[2])

        movie_id_set.add(movie_id)
        user_id_set.add(user_id)
        ratings.append((user_id, movie_id, rating))
        ratings_count_user[user_id] += 1
        ratings_count_movie[movie_id] += 1

    user_id_set = [user_id for user_id in user_id_set if ratings_count_user[user_id] > min_ratings_user]
    movie_id_set = [movie_id for movie_id in movie_id_set if ratings_count_movie[movie_id] > min_ratings_movie]

    A = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)

    # Learn a low rank approximation of the matrix
    A_lr = lrmc(A, low_rank)
    # Then produce an estimate ranking over all items
    return rankings_from_completed_matrix(A_lr)


def ml_1m(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None, preload=True):
    if preload:
        preload_path = "datasets/ratings_datasets/ml_1m_n=250_m=6039_preload.pkl"
        preload_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), preload_path)
        if os.path.exists(preload_path):
            print("Preloaded dataset exists .. loading from there ")
            all_rankings = th.load(preload_path)
            all_items = list(range(len(all_rankings[0])))
            if num_movies < len(all_items):
                all_rankings = extract_sub_rankings(all_rankings, len(all_items), num_movies, seed)
            return list(range(len(all_rankings[0]))), all_rankings
        
    path = "datasets/ratings_datasets/ml-1m-ratings.dat"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    f.readline()
    ratings = []
    movie_id_set = set()
    user_id_set = set()

    ratings_count_user = collections.defaultdict(int)
    ratings_count_movie = collections.defaultdict(int)

    for line in f:
        data = line.rstrip().split('::') # Note the separator
        user_id = int(data[0])
        movie_id = int(data[1])
        rating = float(data[2])

        movie_id_set.add(movie_id)
        user_id_set.add(user_id)
        ratings.append((user_id, movie_id, rating))
        ratings_count_user[user_id] += 1
        ratings_count_movie[movie_id] += 1

    user_id_set = [user_id for user_id in user_id_set if ratings_count_user[user_id] > min_ratings_user]
    movie_id_set = [movie_id for movie_id in movie_id_set if ratings_count_movie[movie_id] > min_ratings_movie]

    A = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)

    # Learn a low rank approximation of the matrix
    A_lr = lrmc(A, low_rank)
    # Then produce an estimate ranking over all items
    return rankings_from_completed_matrix(A_lr)



def ml_10m(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None, preload=True):
    if preload:
        preload_path = "datasets/ratings_datasets/ml_10m_n=300_m=69620_preload.pkl"
        preload_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), preload_path)
        if os.path.exists(preload_path):
            print("Preloaded dataset exists .. loading from there ")
            all_rankings = th.load(preload_path)
            all_items = list(range(len(all_rankings[0])))
            if num_movies < len(all_items):
                all_rankings = extract_sub_rankings(all_rankings, len(all_items), num_movies, seed)
            return list(range(len(all_rankings[0]))), all_rankings
    
    
    path = "datasets/ratings_datasets/ml-10m-ratings.dat"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    f.readline()
    ratings = []
    movie_id_set = set()
    user_id_set = set()

    ratings_count_user = collections.defaultdict(int)
    ratings_count_movie = collections.defaultdict(int)

    for line in f:
        data = line.rstrip().split('::') # Note the separator
        user_id = int(data[0])
        movie_id = int(data[1])
        rating = float(data[2])

        movie_id_set.add(movie_id)
        user_id_set.add(user_id)
        ratings.append((user_id, movie_id, rating))
        ratings_count_user[user_id] += 1
        ratings_count_movie[movie_id] += 1

    user_id_set = [user_id for user_id in user_id_set if ratings_count_user[user_id] > min_ratings_user]
    movie_id_set = [movie_id for movie_id in movie_id_set if ratings_count_movie[movie_id] > min_ratings_movie]

    A = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)

    # Learn a low rank approximation of the matrix
    A_lr = lrmc(A, low_rank)
    # Then produce an estimate ranking over all items
    return rankings_from_completed_matrix(A_lr)


def ml_20m(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None):
    # TODO: load ranking data as well (ranked data determined by ratings)
    path = "datasets/ratings_datasets/ml-20m-ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    f.readline()
    ratings = []
    movie_id_set = set()
    user_id_set = set()

    ratings_count_user = collections.defaultdict(int)
    ratings_count_movie = collections.defaultdict(int)

    for line in f:
        data = line.rstrip().split(',') # Note the seperator
        user_id = int(data[0])
        movie_id = int(data[1])
        rating = float(data[2])

        movie_id_set.add(movie_id)
        user_id_set.add(user_id)
        ratings.append((user_id, movie_id, rating))
        ratings_count_user[user_id] += 1
        ratings_count_movie[movie_id] += 1

    user_id_set = [user_id for user_id in user_id_set if ratings_count_user[user_id] > min_ratings_user]
    movie_id_set = [movie_id for movie_id in movie_id_set if ratings_count_movie[movie_id] > min_ratings_movie]

    A = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)

    # Learn a low rank approximation of the matrix
    A_lr = lrmc(A, low_rank)
    # Then produce an estimate ranking over all items
    return rankings_from_completed_matrix(A_lr)


def book_genome(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None):
    # {"item_id": 41335427, "user_id": 0, "rating": 5}
    path = "datasets/ratings_datasets/book-genome-ratings.json"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    ratings = []
    movie_id_set = set()
    user_id_set = set()

    ratings_count_user = collections.defaultdict(int)
    ratings_count_movie = collections.defaultdict(int)

    for line in f:
        # data = line.rstrip().split(',') # Note the seperator
        data = ast.literal_eval(line.rstrip())

        user_id = data["user_id"]
        movie_id = data["item_id"]
        rating = float(data["rating"])

        movie_id_set.add(movie_id)
        user_id_set.add(user_id)
        ratings.append((user_id, movie_id, rating))
        ratings_count_user[user_id] += 1
        ratings_count_movie[movie_id] += 1
        
    user_id_set = [user_id for user_id in user_id_set if ratings_count_user[user_id] > min_ratings_user]
    movie_id_set = [movie_id for movie_id in movie_id_set if ratings_count_movie[movie_id] > min_ratings_movie]

    A = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)

    # Learn a low rank approximation of the matrix
    A_lr = lrmc(A, low_rank)
    # Then produce an estimate ranking over all items
    return rankings_from_completed_matrix(A_lr)