import numpy as np
import os
from itertools import combinations
from scipy.special import comb
import random
import collections


MOVIES_RATINGS_CUTOFF_LOW = 10
MOVIES_RATINGS_CUTOFF = 25
MOVIES_RATINGS_CUTOFF_HIGH = 50

MOVIES_RATINGS_CUTOFF_ML20M = 50
USERS_RATINGS_CUTOFF_ML20M = 40

BOOK_GENOME_RATINGS_CUTOFF = 50
USERS_RATINGS_CUTOFF_GENOME = 25

TOP_K_CUTOFF = 100
TOP_K_CUTOFF_BOOK = 250


def construct_pairwise_preference_probs(all_rankings, n):
    P = np.eye(n) * 1./2
    comparisons = rank_break(all_rankings)
    for winner, loser in comparisons:
        P[loser, winner] += 1

    for i in range(n - 1):
        for j in range(i+1, n):
            num_comps = P[i, j] + P[j, i]
            if num_comps > 0:
                P[i, j] = P[i, j]/num_comps
                P[j, i] = P[j, i]/num_comps

    return P


def rank_break(ranks):
    """
    Convert a rank dataset into pairwise comparison dataset
    :param ranks:
    :return: List of ((item1, item2), winner) pairwise comparisons
        and preferred item
    """
    Y = []

    for rank in ranks:
        for idx, i in enumerate(rank[:-1]):
            for j in rank[idx+1:]:
                Y.append((i, j)) # i beats j
    return Y


def topological_sort(G: dict, pi_est: list) -> list:
    """
    :param pi_est:
    :param G: dict
    :return:
    """
    def dfs(node: int, visited_nodes: set, unvisited_nodes: list,
            top_order: list):
        visited_nodes.add(node)
        unvisited_nodes.remove(node)
        neighbors = G[node]
        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                dfs(neighbor, visited_nodes, unvisited_nodes, top_order)
        top_order.append(node)

    n = len(pi_est)
    visited = set()
    unvisited = [i for i in pi_est]
    order = []

    while len(visited) < n:
        candidate = list(unvisited)[0]
        dfs(candidate, visited, unvisited, order)

    return list(reversed(order))


def construct_preference_graph(P):
    n = P.shape[0]
    G = dict([(i, list()) for i in range(n)])
    for i in range(n-1):
        for j in range(i+1, n):
            if P[i, j] < 1./2: # j is 'worse' than i
                G[i].append(j)
            else:
                G[j].append(i)
    return G


def estimate_true_ordering(all_ranks):
    # Both APA and sushi exhibit weak stochastic transitivity, so one
    # can estimate the true ordering
    n = len(all_ranks[0])
    pairwise_comparisons = rank_break(all_ranks)

    P = np.eye(n) * 0.5
    for (winner, loser) in pairwise_comparisons:
        P[loser, winner] += 1

    # Compute pairwise preference probabilities
    for i in range(n-1):
        for j in range(i+1, n):
            n_compare = P[i, j] + P[j, i]
            if n_compare == 0:
                continue
            P[i, j] /= n_compare
            P[j, i] /= n_compare

    G = construct_preference_graph(P)
    true_pi = topological_sort(G, list(range(n)))
    return true_pi


def sushi(seed=None):
    all_rankings = []
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "datasets/sushi/ED-00014-00000001.soc")
    
    f = open(filename, 'r')
    num_item = int(f.readline().rstrip())
    for _ in range(num_item + 1): # Skip through the header
        f.readline()
    f.readline() # Then skip the summary
        
    for line in f.readlines():
        line_data = [int(x)-1 for x in line.rstrip().split(",")]
        repeat = line_data[0]+1
        data = line_data[1:]
        if len(data) != 10:
            print(data)
            print(line)
            assert(False)
        for _ in range(repeat):
            all_rankings.append([r-1 for r in data])

    true_ordering = estimate_true_ordering(all_rankings)
    return true_ordering, all_rankings


def apa(year=1,seed=None):
    assert(year <= 12)
    file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__))
                            , "datasets/apa/")
    np.random.seed(seed)
    
    filename = os.path.join(file_directory, f"ED-00028-000000{year}.toc" if year > 9 else f"ED-00028-0000000{year}.toc")
    all_ranks = []
    
    with open(filename, "r") as f:
        num_items = int(f.readline().rstrip())
        for i in range(num_items):  # Read all the candidates
            f.readline()

        f.readline()  # Read the summary line

        # Each line after that:
        # number_of_votes, item, ... {partial}
        for line in f.readlines():
            data = line.rstrip()
            unordered_bottoms = []
            if "{" in data:
            
                partial_rank = data[data.find("{")+1:data.find("}")]
                unordered_bottoms = [int(x)
                                for x in partial_rank.rstrip().split(",")
                                if x.isnumeric()]
            
                ordered_rank = data[:data.find("{")]
            else:
                ordered_rank = data

            ordered_rank = [int(x)
                            for x in ordered_rank.rstrip().split(",")
                            if x.isnumeric()]

            num_votes = ordered_rank[0]
            ordered_rank = ordered_rank[1:]
            # For each partial ranking, generate a linear extension by randomly sample from the bottom
            for _ in range(num_votes):
                linear_extension = ordered_rank + list(random.sample(unordered_bottoms, len(unordered_bottoms)))
                if (len(linear_extension) != 5):
                    print(ordered_rank, unordered_bottoms)
                    assert(False)
                all_ranks.append(linear_extension)
    true_ordering = estimate_true_ordering(all_ranks)
    return true_ordering, all_ranks


def irish(election="north", seed=None):
    if election == "north":
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__))
                                , "datasets/irish/ED-00001-00000001.toc")
    elif election == "west":
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__))
                                , "datasets/irish/ED-00001-00000002.toc")
    elif election == "meath":
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__))
                                , "datasets/irish/ED-00001-00000003.toc")
    else:
        print("Unknown election")
        assert False

    all_ranks = []
    np.random.seed(seed)
    with open(filename, "r") as f:
        # First line is the number of candidates
        num_items = int(f.readline().rstrip())
        for i in range(num_items):  # Read all the candidates
            f.readline()

        f.readline()  # Read the summary line

        # Each line after that:
        # number_of_votes, item, ... {partial}
        for line in f:
            data = line.rstrip()
            unordered_bottoms = []
            if "{" in data:
            
                partial_rank = data[data.find("{")+1:data.find("}")]
                unordered_bottoms = [int(x)-1
                                for x in partial_rank.rstrip().split(",")
                                if x.isnumeric()]
            
                ordered_rank = data[:data.find("{")]
            else:
                ordered_rank = data

            ordered_rank = [int(x)-1
                            for x in ordered_rank.rstrip().split(",")
                            if x.isnumeric()]
                            
            num_votes = ordered_rank[0]+1
            ordered_rank = ordered_rank[1:]
            
            for _ in range(num_votes):
                linear_extension = ordered_rank + list(random.sample(unordered_bottoms, len(unordered_bottoms)))
                if len(linear_extension) != num_items:
                    assert(False)
                all_ranks.append(linear_extension)
    
    true_order = estimate_true_ordering(all_ranks)
    return true_order, all_ranks


def f1(seed=None):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__))
                            , "datasets/f1/ED-00010-00000048.toc")
    all_ranks = []
    np.random.seed(seed)

    with open(filename, "r") as f:
        # First line is the number of candidates
        num_items = int(f.readline().rstrip())
        for i in range(num_items):  # Read all the candidates
            f.readline()

        f.readline()  # Read the summary line

        # Each line after that:
        # number_of_votes, item, ... {partial}
        for line in f:
            data = line.rstrip()
            unordered_bottoms = []
            if "{" in data:
            
                partial_rank = data[data.find("{")+1:data.find("}")]
                unordered_bottoms = [int(x)-1
                                for x in partial_rank.rstrip().split(",")
                                if x.isnumeric()]
            
                ordered_rank = data[:data.find("{")]
            else:
                ordered_rank = data

            ordered_rank = [int(x)-1
                            for x in ordered_rank.rstrip().split(",")
                            if x.isnumeric()]
            
            num_votes = ordered_rank[0]+1
            ordered_rank = ordered_rank[1:]
            
            for _ in range(num_votes):
                linear_extension = ordered_rank + list(random.sample(unordered_bottoms, len(unordered_bottoms)))
                if len(linear_extension) != num_items:
                    assert(False)
                all_ranks.append(linear_extension)


    true_order = estimate_true_ordering(all_ranks)
    return true_order, all_ranks



########################  MOVIES RATINGS DATASETS ####################################

def remove_low_ratings_top_items(ratings, cutoff):
    ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in ratings if num_ratings > cutoff
        ]
    return ratings


def ml_100k(return_rankings=True, cutoff=MOVIES_RATINGS_CUTOFF_LOW, top_k_cutoff=TOP_K_CUTOFF):
    path = "datasets/ml-100k-ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split(',') # Note the separator
        user_id = int(data[0])-1
        movie_id = int(data[1])-1
        movie_id_set.add(movie_id)
        rating = float(data[2])
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
    
    # Collect all the rankings from the users 

    # How can we generate partial rankings of smaller menu size from already partial ranking 


def generate_partial_rankings_from_full_rankings(all_rankings, menus, m=200, seed=None):
    np.random.seed(seed)
    all_partial_rankings = []

    for menu in menus:
        # Randomly pick some m rankings from all rankings (with replacement)
        idx = np.random.choice(len(all_rankings), size=(m,))
        sampled_rankings = [all_rankings[ind] for ind in idx]
        # Extract the items from these rankings while preserving their ordering

        sampled_partial_rankings = []
        for ranking in sampled_rankings:
            sampled_partial_ranking = [x for x in ranking if (x in menu)]
            sampled_partial_rankings.append(sampled_partial_ranking)

        all_partial_rankings.extend(sampled_partial_rankings)

    return all_partial_rankings