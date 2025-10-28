import unittest
from .base_algorithm import BaseAlgorithm
from typing import List
import numpy as np

class CoverTree(BaseAlgorithm):
    """
    Covertree algorithm implementation
    Covertree is a graph-based algorithm that uses the ambient metric to
    construct a heirarchical tree-basec covering structure.
    """

    class Node:
        """
        Auxilary class defining the nodes needed for this tree-based datastructure
        """
        def __init__(self, value: np.ndarray):
            self.value = value
            self.children = []
    
    def __init__(self, name: str, dimension: int, metric: str = "l2", **kwargs):
        super().__init__(name, dimension, **kwargs)
        self.metric = metric
        self.index = None
        self.root = None # root node of index
        
        self.top_level = 0
        
        self.config.update({
            'metric': self.metric
        })

    def build_index(self, vectors: np.ndarray):
        if vectors is None or len(vectors) == 0:
            return
        if self.root is None:
            self.root = self.Node(vectors[0])
            self.max_level = 0
            start_index = 1  
        else:
            start_index = 0  
            
        for i in range(start_index, len(vectors)):
            self.insert(vectors[i])


    def insert(self, p: np.ndarray):
        if self.root is None:
            self.root = self.Node(p)
            self.max_level = 0
            return
        
        Q = [self.root]
        i = self.max_level

        while True:
            if self._insert(p, Q, i):
                break
            else:
                new_root = self.Node(self.root.value)
                new_root.children.append(self.root)
                self.root = new_root
                i += 1
                self.max_level = i
                Q = [self.root]

    def _insert(self, p: np.ndarray, Q: List[Node], level: int) -> bool:
        """
        Refer to Algorithm 2 (Insert) in Bey26 for a reference
        """
        dist_power = 2 ** level

        # 1. Set Q_children
        Q_children = []
        for q in Qi:
            Q_children.extend(q.children)
            
        # 3. (a) Set Qi_minus_1 (the filtered set)
        Qi_minus_1 = [
            child for child in Q_children 
            if np.linalg.norm(p - child.value) <= dist_power
        ]

        # 3. (b) if Insert(p, Qi−1, i − 1) ...
        if Qi_minus_1:
            if self._insert(p, Qi_minus_1, level - 1):
                return True # "parent found"

        # 3. (b) ... and d(p, Qi) ≤ 2i ...
        for q in Qi:
            if np.linalg.norm(p - q.value) <= dist_power:
                q.children.append(self.Node(p))
                return True # "parent found"
        
        # 3. (c) else return “no parent found”
        return False

    def search(self, p: np.ndarray) -> np.ndarray | None:
        """
        Finds the nearest neighbor to point p (Algorithm 1: Find-Nearest).
        """
        if self.root is None:
            return None

        # Start with the root as the best-so-far
        best_q_node = self.root
        best_dist = np.linalg.norm(p - self.root.value)
        
        # Q_i is our set of candidate nodes for the current level
        Q_i = [self.root]
        i = self.max_level

        while i >= -np.inf: # Loop "down to -infinity"
            # 1. Set Q_children
            Q_children = []
            for q_node in Q_i:
                Q_children.extend(q_node.children)

            if not Q_children:
                break # We've hit the bottom of the tree

            # Find the minimum distance to any child
            child_dists = [np.linalg.norm(p - c.value) for c in Q_children]
            min_child_dist = np.min(child_dists)
            min_child_index = np.argmin(child_dists)

            # Update our global best-so-far
            if min_child_dist < best_dist:
                best_dist = min_child_dist
                best_q_node = Q_children[min_child_index]

            # 2. (b) Form the next cover set, Q_i-1
            # d(p, Q_children) is min_child_dist
            threshold = min_child_dist + (2 ** i)
            
            Q_next = []
            for j in range(len(Q_children)):
                if child_dists[j] <= threshold:
                    Q_next.append(Q_children[j])

            if not Q_next:
                break # No children qualified, we're done

            Q_i = Q_next
            i -= 1
        
        return best_q_node.value

    def __str__(self):
        P = [self.root]
        i = self.max_level
        return_str = ""
        while P:
            return_str += "\n"
            Q = []
            p = P.pop()
            level_str = "i:"
            for p in P:
                level_str += f" {p.value}"
                Q.extend(p.children)
            return_str += level_str
            P = Q
        return return_str
# --- Unit Test Class ---

class TestCoverTreeSearch(unittest.TestCase):
    
    def setUp(self):
        self.tree = CoverTree(name="test_search_tree", dimension=2)

    def test_search_empty_tree(self):
        """Test searching an empty tree."""
        query = np.array([1, 1])
        result = self.tree.search(query)
        self.assertIsNone(result)

    def test_search_single_node_tree(self):
        """Test searching a tree with just one node."""
        p1 = np.array([5, 5])
        self.tree.insert(p1)
        
        query = np.array([1, 1])
        result = self.tree.search(query)
        np.testing.assert_array_equal(result, p1)

    def test_search_exact_match(self):
        """Test searching for a point that is already in the tree."""
        p1 = np.array([0, 0])
        p2 = np.array([5, 5])
        self.tree.build_index(np.array([p1, p2]))
        
        query = np.array([5, 5])
        result = self.tree.search(query)
        np.testing.assert_array_equal(result, p2)

    def test_search_nearest_neighbor_simple(self):
        """Test finding the correct nearest neighbor in a simple tree."""
        p1 = np.array([0, 0])
        p2 = np.array([5, 5])
        self.tree.build_index(np.array([p1, p2]))
        
        # Query point is clearly closer to p1
        query = np.array([0.1, 0.1])
        result = self.tree.search(query)
        np.testing.assert_array_equal(result, p1)
        
        # Query point is clearly closer to p2
        query = np.array([4.9, 4.9])
        result = self.tree.search(query)
        np.testing.assert_array_equal(result, p2)

    def test_search_after_level_growth(self):
        """Test search after an insert forced the max_level to grow."""
        p1 = np.array([0, 0])
        p2 = np.array([10, 10]) # This will force max_level to 4
        self.tree.build_index(np.array([p1, p2]))
        
        self.assertEqual(self.tree.max_level, 4) # Verify tree structure
        
        # Query near p2
        query_near_p2 = np.array([9, 9])
        result = self.tree.search(query_near_p2)
        np.testing.assert_array_equal(result, p2)
        
        # Query near p1
        query_near_p1 = np.array([1, 1])
        result = self.tree.search(query_near_p1)
        np.testing.assert_array_equal(result, p1)

    def test_search_complex_cluster(self):
        """Test search in a more complex tree."""
        points = np.array([
            [0, 0],    # p1
            [0.1, 0.1],# p2 (child of p1)
            [10, 10],  # p3 (far away)
            [10.1, 10],# p4 (child of p3)
            [5, 5]     # p5 (mid-point)
        ])
        self.tree.build_index(points)
        
        # Query near [0, 0] cluster
        query1 = np.array([-0.1, 0])
        result1 = self.tree.search(query1)
        np.testing.assert_array_equal(result1, np.array([0, 0]))
        
        # Query near [10, 10] cluster
        query2 = np.array([10.2, 10.2])
        result2 = self.tree.search(query2)
        np.testing.assert_array_equal(result2, np.array([10.1, 10]))
        
        # Query near the mid-point
        query3 = np.array([4.5, 4.5])
        result3 = self.tree.search(query3)
        np.testing.assert_array_equal(result3, np.array([5, 5]))


if __name__ == '__main__':
    unittest.main()




