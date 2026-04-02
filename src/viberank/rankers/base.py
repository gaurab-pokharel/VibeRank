import numpy as np
import choix
import random
from abc import ABC, abstractmethod


class Ranker(ABC):
    def __init__(self, items, comparator, seed=None):
        """
        Abstract base class for ranking algorithms.

        Args:
            items (list): List of items to be ranked.
            comparator (object): Comparator used for noisy comparisons.
            seed (int, optional): Random seed for reproducibility.
                                  Ensures that the same pairs are chosen every time if provided.
        """
        self.items = items
        self.N = len(items)
        self.comparator = comparator
        self.seed = seed
    
        self.total_samples = 0       # Total number of comparisons made.
        self.learned_ranking = None  # Final aggregated ranking (e.g., using Rank Centrality).
        self.bt_ranking = None       # Ranking using the Bradley–Terry model.
        self.win_matrix = None       # Store the win-matrix returned by the comparator

        # Set the random seed for reproducibility if provided.
        if seed is not None:
            random.seed(seed)        # Seed Python's random module.
            np.random.seed(seed)     # Seed NumPy's random number generator.

    @abstractmethod
    def run(self):
        """
        Abstract method to run the ranking algorithm.
        """
        pass

    def get_bt_ranking(self, alpha=0.01, max_iter=1000):
        """
        Compute and return item rankings based on the Bradley–Terry model.

        Args:
            alpha (float): Regularization parameter for the Choix optimization.
            max_iter (int): Maximum number of iterations allowed in the BT model optimization.

        Requires: 
            self.win_matrix: a two-dimensional (NxN) NumPy array, where N is the number of items being ranked. 
            For any two items i and j, the entry at win_matrix[i, j] represents the number of times item i won
            in a comparison against item j.
            
        Returns:
            list: Items ranked from most to least preferred.
        """
        
        # Ensure the win matrix exists.
        if not hasattr(self, 'win_matrix'):
            raise ValueError("The win matrix is not available. Please run the ranking algorithm first.")

        if self.win_matrix is None: 
            raise ValueError('Win Matrix hasn not been initialized. Make sure it has been initialized.')
        
        comparisons_bt = []
        # Build the list of comparisons (each pair repeated by its win count).
        for i in range(self.N):
            for j in range(self.N):
                count = self.win_matrix[i, j]
                if count > 0:
                    comparisons_bt.extend([(i, j)] * int(count))
        
        # Fit the Bradley–Terry model using Choix.
        ratings_bt = choix.ilsr_pairwise(self.N, comparisons_bt, alpha=alpha, max_iter=max_iter)
        # Sort items by their ratings (higher rating means better rank).
        ranked_indices = np.argsort(-np.array(ratings_bt))
        self.bt_ranking = [self.comparator.get_item(idx) for idx in ranked_indices]
        # Update total samples based on the win matrix.
        self.total_samples = int(np.sum(self.win_matrix))
        return self.bt_ranking