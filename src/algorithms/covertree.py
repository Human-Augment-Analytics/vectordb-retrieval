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
        pass

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
        if not Q:
            return False
        
        candidates = []
        for q in Q:
            candidates.extend(q.children)

        if (
            not candidates
            or np.min([np.linalg.norm(p - c.value) for c in candidates]) > 2 ** level
        ):
            return False
        
        if (
            not self._insert(p, candidates, level - 1)
            and np.min([np.linalg.norm(p - q.value) for q in Q]) <= 2 ** level
        ):
            # Might want to experiment with doing this randomly so as to create
            # a more balanced tree
            for q in Q:
                if np.linalg.norm(p - q.value) <= 2 ** level:
                    q.children.append(self.Node(p))
                    return True

        return False

    def search(self):
        pass

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





