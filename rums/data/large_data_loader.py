import numpy as np
import scipy as sp
import os
import collections
import nimfa


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



def ml_100k(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None):
    # TODO: load ranking data as well (ranked data determined by ratings)
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


def ml_1m(seed=None, min_ratings_user=10, min_ratings_movie=10, low_rank=30, num_users=None, num_movies=None):
    # TODO: load ranking data as well (ranked data determined by ratings)
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