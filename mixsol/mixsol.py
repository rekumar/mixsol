import numpy as np
import networkx as nx
from scipy.optimize import nnls
from mixsol.solution import Solution


class SolutionMixer:
    def __init__(self, stock_solutions: list, targets: dict = {}):
        self.target_solutions = list(targets.keys())
        self.target_volumes = targets  # dict of solution:volume
        self.solutions = stock_solutions + self.target_solutions
        (
            self.solution_matrix,
            self.solvent_idx,
            self.components,
        ) = self._solutions_to_matrix(stock_solutions + self.target_solutions)
        self.stock_idx = [
            i for i in range(len(stock_solutions))
        ]  # rows of solution matrix that belong to stock solutions

    def _solutions_to_matrix(self, solutions: list, components: list = None):
        """Converts a list of solutions into a matrix of molarity of each component in each solution

        Args:
            solutions (list): Solution objects
            components (list, optional): individual components that cover all solutions. If none, will generate this from the solution list

        Returns:
            solution_matrix, solvent_idx, components

            solution_matrix = rows are solutions, columns are components

        """
        if isinstance(solutions, Solution):
            solutions = [solutions]

        # get possible solution components from stock list
        if components is None:
            components = set()
            for s in solutions:
                components.update(s.solute_dict.keys(), s.solvent_dict.keys())
            components = list(
                components
            )  # sets are not order-preserving, lists are - just safer this way

        # organize components into a stock matrix, keep track of which rows are solvents
        solution_matrix = np.zeros((len(solutions), len(components)))
        solvent_idx = set()
        for m, s in enumerate(solutions):
            for n, c in enumerate(components):
                if c in s.solute_dict:
                    solution_matrix[m, n] = s.solute_dict[c] * s.molarity
                elif c in s.solvent_dict:
                    solution_matrix[m, n] = s.solvent_dict[c]
                    solvent_idx.add(m)
        solvent_idx = list(solvent_idx)

        return solution_matrix, solvent_idx, components

    def _solution_to_vector(self, target: Solution, volume=1):
        # organize target solution into a matrix of total mols desired of each component
        target_matrix = np.zeros((len(self.components),))
        for m, c in enumerate(self.components):
            if c in target.solute_dict:
                target_matrix[m] = target.solute_dict[c] * target.molarity * volume
            elif c in target.solvent_dict:
                target_matrix[m] = target.solvent_dict[c] * volume
        return target_matrix.T

    def _calculate_mix(
        self, target, solution_indices: list = None, tolerance=1e-3, min_fraction=0
    ):
        if solution_indices is None:
            solution_indices = list(range(len(self.solutions)))
        A = self.solution_matrix[solution_indices]
        b = self._solution_to_vector(target)
        x, err = nnls(A.T, b, maxiter=1e3)
        x[x < 1e-10] = 0
        if err > tolerance:
            return np.nan
        if np.logical_and(x > 0, x < min_fraction).any():
            return np.nan

        x_full = np.zeros((len(self.solutions),))
        for idx, x_ in zip(solution_indices, x):
            x_full[idx] = x_
        return x_full

    def mix(
        self,
        target,
        volume,
        solution_indices=None,
        tolerance=1e-3,
        min_volume=0,
        verbose=False,
    ):
        min_fraction = min_volume / volume
        if solution_indices is None or solution_indices == "stock":
            solution_indices = self.stock_idx
        elif solution_indices == "all":
            solution_indices = list(range(len(self.solutions)))

        possible_mixtures = []
        for i in range(1, len(self.solutions)):
            for idx in itt.combinations(solution_indices, i):
                x = self._calculate_mix(
                    target,
                    solution_indices=list(idx),
                    tolerance=tolerance,
                    min_fraction=min_fraction,
                )
                if not np.isnan(x).any():
                    possible_mixtures.append(x)
        if len(possible_mixtures) == 0:
            return np.nan
        possible_mixtures.sort(
            key=lambda x: -min(x[x > 0])
        )  # sort such that the mixture with the largest minimum fraction is first
        if verbose:
            return possible_mixtures * volume
        return possible_mixtures[0] * volume

    def mixture_graph(self, min_volume=20):
        self.graph = np.zeros((len(self.solutions), len(self.solutions)))
        self.availableidx = self.stock_idx.copy()
        # self.graph[np.diag_indices(len(self.solutions))] = 1
        attempts = 0
        while len(self.availableidx) < len(self.solutions):
            if attempts > 5:
                print("Could not find a solution")
                break
            for i in range(len(self.solutions)):
                if i in self.availableidx:
                    continue
                x = self.mix(
                    target=self.solutions[i],
                    volume=self.target_volumes[self.solutions[i]],
                    solution_indices=self.availableidx,
                    min_volume=min_volume,
                )
                if not np.isnan(x).any():
                    self.graph[i, :] = x
                    self.availableidx.append(i)
            attempts += 1
        # return self.graph
