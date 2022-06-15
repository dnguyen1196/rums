import collections
import numpy as np


class GeneralizedBordaScore():
    def __init__(self, n):
        self.n = n
        self.ordering = None

    def __str__(self):
        return "BC"

    def fit(self, partial_rankings):
        wins = collections.defaultdict(int)

        for ranking in partial_rankings:
            num = len(ranking)
            for i, item in enumerate(ranking):
                wins[item] += num - i - 1

        counts = [(item, wins) for item, wins in wins.items()]
        sorted_counts = sorted(counts, key=lambda x: x[1], reverse=True)
        self.ordering = [item for item, _ in sorted_counts]
        return self.ordering
