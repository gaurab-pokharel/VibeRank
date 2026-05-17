from viberank.rankers.base import Ranker 
import random 
import numpy as np 
import os 
import json 

class RC(Ranker):
    """
    RC: Rank Centrality-based Ranking Class
    =========================================

    The RC class is a concrete implementation of the abstract Ranker class. It
    implements a ranking algorithm based on pairwise comparisons using the Rank
    Centrality method. The primary workflow involves generating a "tie sheet" of
    item pairs that are to be compared, obtaining the corresponding win matrix
    from an external comparator, and finally computing a ranking based on the
    results of these comparisons.

    Initialization Parameters:
    --------------------------
    items : list
        A list of items to be ranked.
    comparator : object
        An object that provides the comparisons between items. The comparator
        is expected to implement:
            - compare_items(tie_sheet): Takes a list of item index pairs and returns a win matrix.
            - get_item(index): Maps an index to the corresponding item.
    compare_probability : float, optional (default: 0.3)
        The probability that a given pair of items will be selected for comparison.
        This controls the sparsity of the tie sheet.
    seed : int, optional
        A random seed value that ensures reproducibility in the selection of pairs.
        If provided, both Python's random module and NumPy's random generator are seeded.

    Attributes:
    -----------
    compare_probability : float
        The probability threshold used when determining which pairs to include in
        the tie sheet.
    tie_sheet : list of tuple
        A list containing pairs of item indices (i, j) that were selected for comparison.
    win_matrix : np.ndarray
        A square matrix of dimensions (N, N) where N is the number of items. The entry
        win_matrix[i, j] indicates the number of times item i won against item j in the
        pairwise comparisons.
    total_samples : int
        The total number of pairwise comparisons made. This is computed as the sum of
        all entries in the win_matrix.
    learned_ranking : list
        The final ordering of items as computed by the Rank Centrality algorithm.
    bt_ranking : list
        The ranking computed via an alternative Bradley–Terry model (if computed).
    
    Methods:
    --------
    make_tie_sheet()
        Generates and returns a tie sheet (list of index pairs) representing which
        pairs of items should be compared. Each unique pair (i, j) is added with the
        probability specified by compare_probability.

    rank_centrality(win_matrix, tol=1e-8, num_iterations=1000)
        Computes ranking scores using the Rank Centrality algorithm. It takes as input
        a win matrix and returns a tuple (p, P) where p is the vector of ranking scores
        (stationary distribution) and P is the constructed transition matrix.

    run()
        Executes the ranking process by generating a tie sheet, obtaining the win matrix
        via the comparator, and then computing the rankings using the Rank Centrality
        algorithm. It also stores the computed tie sheet, win matrix, and total number of
        comparisons.

    get_tie_sheet()
        Returns the tie sheet generated during the run.

    get_win_matrix()
        Returns the win matrix obtained from the comparator.

    get_total_comapriosons()
        Returns the total number of comparisons performed (i.e., the sum of the win matrix).

    Note:
    -----
    - The 'comparator' should implement a method called 'compare_items' that accepts a tie sheet
      (a list of tuple pairs) and returns the win matrix.
    - The 'comparator' is also expected to have a 'get_item' method to translate indices back to
      original item representations.
    - The run method accepts a compare_probability argument. It is generally recommended that this
      value match the one provided upon initialization.
    """

    def __init__(self, items, comparator, compare_probability=0.3, seed=None, folder_name=None):
        super().__init__(items, comparator, seed)
        self.compare_probability = compare_probability
        self.folder_name = folder_name
    
    def set_tie_sheet_path(self,path):
        self.stored_sheet_path = path

    def make_tie_sheet(self):
        """
        Generate a tie sheet (list of index pairs) representing which pairs of items should be compared.
        
        The method iterates over all unique unordered pairs (i, j) of items (where i < j) and adds
        the pair to the tie sheet with a probability defined by self.compare_probability.
        
        Returns:
            list of tuple: List of index pairs (i, j) selected for comparison.
        """
        filename = os.path.join(self.folder_name, f"tie_sheet.json")
        if os.path.exists(filename): 
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else: 
            tie_sheet = []
            # Loop over all unique unordered pairs of items.
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    # With the given probability, add the pair (i, j) to the tie sheet.
                    if random.random() < self.compare_probability:
                        tie_sheet.append((i, j))
            return tie_sheet
    
    def rank_centrality(self, win_matrix, tol=1e-8, num_iterations=1000):
        """
        Compute ranking scores using the Rank Centrality algorithm based on a win matrix.

        This method constructs a transition matrix from the win matrix by computing for each pair (i, j)
        the fraction of wins of j over the total number of comparisons between i and j. It then uses
        power iteration to compute the stationary distribution which represents the ranking scores.

        Args:
            win_matrix (np.ndarray): A matrix where win_matrix[i, j] is the number of wins of item i over j.
            tol (float): Tolerance level for convergence in the power iteration loop.
            num_iterations (int): Maximum number of iterations allowed for the power iteration.

        Returns:
            tuple: A tuple (p, P) where:
                - p (np.ndarray): The stationary distribution vector (ranking scores) for each item.
                - P (np.ndarray): The transition matrix constructed from the win_matrix.
        """
        N = win_matrix.shape[0]
        A = np.zeros((N, N))

        # Calculate the fraction of wins of j over i for each pair (i, j).
        for i in range(N):
            for j in range(N):
                if i != j:
                    total = win_matrix[i, j] + win_matrix[j, i]
                    if total > 0:
                        A[i, j] = win_matrix[j, i] / total
                    else:
                        A[i, j] = 0.0

        # Compute "out-degree" for each item.
        d = np.sum(A, axis=1)
        d_max = np.max(d)

        # Construct the transition matrix P.
        P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    P[i, j] = A[i, j] / d_max
            # Self-loop probability to ensure rows sum to one.
            P[i, i] = 1 - np.sum(A[i, :]) / d_max

        # Power iteration to determine the stationary distribution.
        p = np.ones(N) / N  # Start with a uniform distribution.
        for _ in range(num_iterations):
            new_p = p.dot(P)
            if np.linalg.norm(new_p - p, 1) < tol:
                p = new_p
                break
            p = new_p
        p = p / np.sum(p)  # Normalize to ensure the vector sums to 1.
        return p, P

    def run(self,offline=False):
        """
        Execute the ranking process.

        The run method carries out the following steps:
          1. Generates a tie sheet of item pairs based on the specified compare_probability.
          2. Passes the tie sheet to the comparator via the 'compare_items' method to obtain the win matrix.
          3. Computes the total number of comparisons made.
          4. Uses the Rank Centrality algorithm to compute ranking scores.
          5. Maps the computed scores back to the original items and sorts them to produce the final ranking.

        Args:
            compare_probability (float): The probability that a given pair of items is compared.
                (For consistency, this should match the compare_probability set during initialization.)

        Returns:
            None
        """
        # Create a tie sheet using the abstracted method.
        if not offline:
            self.tie_sheet = self.make_tie_sheet()
            self._store_tie_sheet()
        else:
            #self.load_tie_sheet()
            self.tie_sheet = self.comparator.make_tie_sheet_from_comparion_data()
        print(self.tie_sheet)
        # Pass the tie sheet to the comparator.
        # The comparator should return a win matrix where win_matrix[i, j] reflects the number of times
        # item i wins over item j.
        self.win_matrix = self.comparator.compare_items(self.tie_sheet)
        self.total_samples = self.comparator.num_comparisons
        
        # Compute ranking scores using the Rank Centrality algorithm.
        ranking_scores, _ = self.rank_centrality(self.win_matrix)
        
        # Map computed scores back to the original items.
        item_scores = {self.comparator.get_item(i): ranking_scores[i] for i in range(self.N)}
        
        # Sort items by descending ranking scores.
        ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        self.learned_ranking = [item[0] for item in ranked_items]

    def get_tie_sheet(self):
        """
        Retrieve the tie sheet generated during the run.

        Returns:
            list of tuple: The list of item index pairs that were compared.
        """
        
        return self.tie_sheet 
    
    def get_win_matrix(self):
        """
        Retrieve the win matrix obtained from the comparator.

        Returns:
            np.ndarray: A square matrix where win_matrix[i, j] indicates the number of wins
                        of item i over item j.
        """
        return self.win_matrix
    
    def get_total_comaprisons(self):
        """
        Retrieve the total number of comparisons made.

        This is computed as the sum of all entries in the win matrix.

        Returns:
            int: The total count of pairwise comparisons.
        """
        return self.total_samples

   
    def load_tie_sheet(self):
        filename = os.path.join(self.stored_sheet_path, f"tie_sheet.json")
        with open(filename, "r") as f:
            self.tie_sheet = json.load(f)
        self.tie_sheet  = [tuple(pair) for pair in self.tie_sheet ]
        print(self.tie_sheet)



    def _store_tie_sheet(self):
        """
        Store the provided tie sheet in the designated folder.

        The tie sheet is written in JSON format with a file name that includes a timestamp.

        Args:
            tie_sheet (list of tuple): The tie sheet to store.
        """
        filename = os.path.join(self.folder_name, f"tie_sheet.json")
        with open(filename, "w") as f:
            json.dump(self.tie_sheet, f)


