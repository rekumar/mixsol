import numpy as np


class DirectedGraph:
    """Given an adjacency matrix, allows for some operations on a directed graph.

    Specifically, this is used to get the hierarchy/order of solution mixing,
    and to backpropagate solution needs for all downstream mixing.

    """

    def __init__(self, adjacency_matrix):
        self.g = adjacency_matrix

    def indegree(self, node: int) -> int:
        """return the number of nodes that are directed in to a given node

        Args:
            node (int): index of the node for which to calculate incoming degree

        Returns:
            int: incoming degree of node
        """
        return sum(self.g[node] > 0)

    def children(self, node: int) -> list:
        """return list of node indices that a given node is outgoing to

        Args:
            node (int): index of the node from which to get outgoing nodes

        Returns:
            list: list of node indices outgoing from given node
        """
        return [n for n, v in enumerate(self.g[:, node]) if v > 0]

    def hierarchy(self, reverse: bool = False) -> list:
        """returns generations of the graph hierarchy

        Args:
            reverse (bool, optional): whether to return inner->outer generations (True) or outer->inner(False). Defaults to False.

        Returns:
            list: nested list of node indices in each generation
        """
        hier = [h for h in self._hierarchy_iter()]
        if reverse:
            return hier[::-1]
        else:
            return hier

    def _hierarchy_iter(self) -> iter:
        """returns an iterator over the hierarchy of the graph (outer->inner)

        Yields:
            list: list of node indices for this generation
        """
        indegree_map = {}
        zero_indegree = []
        for node in range(self.g.shape[0]):
            d = self.indegree(node)
            if d > 0:
                indegree_map[node] = d
            else:
                zero_indegree.append(node)

        while zero_indegree:
            this_generation = zero_indegree
            zero_indegree = []
            for node in this_generation:
                for child in self.children(node):
                    indegree_map[child] -= 1
                    if indegree_map[child] == 0:
                        zero_indegree.append(child)
                        del indegree_map[child]
            yield this_generation

    def _normalize(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """normalizes the adjacency matrix so each row (incoming edges) sums to 1"""
        v_in = adjacency_matrix.sum(axis=1)
        selfidx = np.where(v_in == 0)[0]
        v_in[selfidx] = 1
        g_norm = adjacency_matrix / v_in[:, np.newaxis]
        g_norm[selfidx, selfidx] = 1
        return g_norm

    def propagate_load(self, load: list) -> list:
        """propagates the final desired load of each node upstream the mixing order.
            returns the initial load of each node needed for downstream mixing, such that
            we end with the desired load at each node.

        Args:
            load (list): list of loads (ending amounts) desired at each node

        Raises:
            ValueError: if the list of loads is a different length than the number of nodes in the graph

        Returns:
            list: list of starting loads (initial amounts) needed for each node to supply enough load to downstream nodes
        """
        if len(load) != self.g.shape[0]:
            raise ValueError("load must have same number of elements as graph nodes!")
        g_norm = self._normalize(self.g)

        first_gen = self.hierarchy()[0]
        for gen in self.hierarchy(reverse=True):
            for node in gen:
                needed = g_norm[node] * load[node]
                if node in first_gen:
                    load[node] = needed[node]
                else:
                    load = load + needed
        return load
