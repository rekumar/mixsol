import numpy as np
from scipy.optimize import nnls
from mixsol.components import Solution, Powder
from mixsol.helpers import *
import itertools as itt
import matplotlib.pyplot as plt


class DirectedGraph:
    def __init__(self, adjacency_matrix):
        self.g = adjacency_matrix

    def indegree(self, node):
        return sum(self.g[node] > 0)

    def children(self, node):
        return [n for n, v in enumerate(self.g[:, node]) if v > 0]

    def hierarchy(self, reverse=False):
        hier = [h for h in self._hierarchy_iter()]
        if reverse:
            return hier[::-1]
        else:
            return hier

    def _hierarchy_iter(self):
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

    def _normalize(self, adjacency_matrix):
        v_in = adjacency_matrix.sum(axis=1)
        selfidx = np.where(v_in == 0)[0]
        v_in[selfidx] = 1
        g_norm = adjacency_matrix / v_in[:, np.newaxis]
        g_norm[selfidx, selfidx] = 1
        return g_norm

    def propagate_load(self, load):
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


class Mixer:
    def __init__(self, stock_solutions: list, targets: dict = None):
        if targets is None:
            targets = {}
        self.target_solutions = list(targets.keys())
        self.solutions = list(set(stock_solutions + self.target_solutions))
        self.target_volumes = {
            solution: targets.get(solution, 0) for solution in self.solutions
        }
        (
            self.solution_matrix,
            self.solvent_idx,
            self.components,
        ) = self._solutions_to_matrix(self.solutions)
        self.stock_idx = [
            self.solutions.index(stock) for stock in stock_solutions
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

    def _is_plausible(self, solution_matrix, target_vector):
        """Check if the solution_matrix spans target vector values. If not, no chance of a successful mixture here!"""
        difference = solution_matrix - target_vector
        return not any([all(v > 0) or all(v < 0) for v in difference.T])

    def _calculate_mix(
        self, target, solution_indices: list = None, tolerance=1e-3, min_fraction=0
    ):
        if solution_indices is None:
            solution_indices = list(range(len(self.solutions)))
        A = self.solution_matrix[solution_indices]
        b = self._solution_to_vector(target)
        if not self._is_plausible(A, b):
            return np.nan
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
        tolerance=1e-2,
        min_volume=0,
        verbose=False,
        max_inputs=None,
    ):
        min_fraction = min_volume / volume
        if solution_indices is None or solution_indices == "stock":
            solution_indices = self.stock_idx
        elif solution_indices == "all":
            solution_indices = list(range(len(self.solutions)))

        possible_mixtures = []
        if max_inputs is None:
            max_inputs = len(self.solutions)
        for i in range(1, max_inputs):
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

    def _solve_adjacency_matrix(self, min_volume, max_inputs, max_generations):
        graph = np.zeros((len(self.solutions), len(self.solutions)))
        self.availableidx = self.stock_idx.copy()
        generation = 0
        n_remaining = len(self.solutions) - len(self.availableidx)
        n_remaining_lastiter = np.inf
        while n_remaining > 0:
            if generation > max_generations or n_remaining == n_remaining_lastiter:
                print("Could not find a solution")
                for i, s in enumerate(self.solutions):
                    if i not in self.availableidx:
                        print(s)
                break
            n_remaining_lastiter = n_remaining

            for i in range(len(self.solutions)):
                if i in self.availableidx:
                    continue
                x = self.mix(
                    target=self.solutions[i],
                    volume=self.target_volumes[self.solutions[i]],
                    solution_indices=self.availableidx,
                    min_volume=min_volume,
                    max_inputs=max_inputs,
                )
                if not np.isnan(x).any():
                    graph[i, :] = x
                    self.availableidx.append(i)
                    n_remaining -= 1
            generation += 1
        return graph

    def solve(self, min_volume, max_inputs=None, max_generations=np.inf):
        if max_inputs is None:
            max_inputs = len(self.solutions) - 1

        adjacency_matrix = self._solve_adjacency_matrix(
            min_volume=min_volume,
            max_inputs=max_inputs,
            max_generations=max_generations,
        )
        self.graph = DirectedGraph(adjacency_matrix)
        g_norm = self.graph._normalize(self.graph.g)
        v_end = np.array([self.target_volumes.get(soln, 0) for soln in self.solutions])
        v_needed = self.graph.propagate_load(v_end)
        self.mixing_order = []
        for generation in self.graph.hierarchy():
            mixes_in_this_gen = {}
            for solution_index in generation:
                this_mix = {}
                for input_solution, input_fraction in zip(
                    self.solutions, g_norm[solution_index]
                ):
                    if input_fraction > 0:
                        this_mix[input_solution] = (
                            v_needed[solution_index] * input_fraction
                        )
                mixes_in_this_gen[self.solutions[solution_index]] = this_mix
            self.mixing_order.append(mixes_in_this_gen)

        transfers = {}
        self.stock_volumes = {}
        for gen in self.mixing_order:
            for node, sources in gen.items():
                for source, volume in sources.items():
                    if source == node:
                        self.stock_volumes[node] = volume
                        continue
                    if source not in transfers:
                        transfers[source] = {}
                    transfers[source][node] = volume
        self.transfers_per_generation = []
        for gen in self.mixing_order:
            this_gen = {node: transfers[node] for node in gen if node in transfers}
            if len(this_gen) > 0:
                self.transfers_per_generation.append(this_gen)

    ### publishing methods
    def print(self):
        # for idx in self.stock_idx:
        #     volume_graph[idx, idx] = self.target_volumes.get(self.solutions[idx], 0)
        print("===== Stock Prep =====")
        for solution, volume in self.stock_volumes.items():
            print(f"{volume:.2f} of {solution}")
        print(f"====== Mixing =====")
        for generation in self.transfers_per_generation:
            for source, transfers in generation.items():
                print(f"Distribute {source}:")
                for destination, volume in transfers.items():
                    print(f"\t{volume:.2f} to {destination}")

    def plot(self):
        nodes = {}
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        for i, gen in enumerate(self.mixing_order):
            xoff = 0.5 * (i % 2) + np.random.random() * 0.2
            for j, (node, sources) in enumerate(gen.items()):
                if node not in nodes:
                    x = j * 1.25 + xoff * len(self.mixing_order)
                    y = i
                    c = plt.cm.Set2(i)
                    nodes[node] = (x, y, c)
                else:
                    x, y, c = nodes[node]
                for source, volume in sources.items():
                    if source == node:
                        x1 = x + 0.1 * len(self.mixing_order)
                        y1 = y
                        rotation = 0
                        ha = "left"
                        _, _, c = nodes[source]
                    else:
                        x1, y1, c = nodes[source]
                        np.rad2deg(np.arctan((y - y1) / (x - x1)))
                        ha = "center"
                        ax.arrow(
                            x=x1,
                            y=y1,
                            dx=x - x1,
                            dy=y - y1,
                            length_includes_head=True,
                            head_width=0.1,
                            color="k",
                            alpha=0.5,
                        )

                    factor = 0.25 + 0.5 * np.random.random()
                    plt.text(
                        x=x + (x1 - x) * factor,
                        y=y + (y1 - y) * factor,
                        s=f"{volume:.2f}",
                        ha=ha,
                        va="center",
                        rotation=rotation,
                        bbox=dict(boxstyle="round", fc=c),
                    )
                plt.scatter(x, y, color=plt.cm.Set2(i), s=300, zorder=999, alpha=0.5)
                plt.text(
                    x=x,
                    y=y,
                    s=str(node),
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=1000,
                )

        plt.tight_layout()


class SolutionMaker:
    def __init__(self, powders: list):
        self.powders = powders
        self.matrix, self.components = self._powders_to_matrix(powders)

    def _powders_to_matrix(self, powders: list):
        # get possible solution components from stock list
        components = set()
        for p in powders:
            components.update(p.components.keys())
        components = list(
            components
        )  # sets are not order-preserving, lists are - just safer this way

        # organize components into a stock matrix, keep track of which rows are solvents
        solid_matrix = np.zeros((len(powders), len(components)))
        for m, p in enumerate(powders):
            for n, c in enumerate(components):
                solid_matrix[m, n] = p.components.get(c, 0)
        return solid_matrix, components

    def _solution_to_vector(self, target: Solution, volume: float = 1):
        # organize target solution into a matrix of total mols desired of each component
        target_matrix = np.zeros((len(self.components),))
        for m, component in enumerate(self.components):
            if component in target.solute_dict:
                target_matrix[m] = (
                    target.solute_dict[component] * target.molarity * volume
                )
        return target_matrix.T

    def _filter_powders(self, target):
        idx_to_use = list(range(self.matrix.shape[0]))
        for component_idx in np.where(target == 0)[0]:
            present_in_powder = np.where(self.matrix[:, component_idx] > 0)[0]
            for powder_idx in present_in_powder:
                idx_to_use.remove(powder_idx)
        return idx_to_use

    def _calculate_mix(self, matrix, target, tolerance=1e-3):
        x, err = nnls(matrix.T, target.T, maxiter=1e5)
        # x[x < 1e-10] = 0
        # if err > tolerance:
        #     return np.nan
        return x, err

    def get_weights(self, target: Solution, volume: float, tolerance=1e-10):
        target_vector = self._solution_to_vector(target=target, volume=volume)
        usable_powder_indices = self._filter_powders(target_vector)
        matrix = self.matrix[usable_powder_indices]
        mass_vector, error = nnls(matrix.T, target_vector.T, maxiter=1e3)
        if error > tolerance:  # TODO #1
            raise Exception(
                f"Could not achieve target solution from given powders. Error={error}, tolerance was {tolerance}"
            )
        return {
            str(self.powders[idx]): m
            for idx, m in zip(usable_powder_indices, mass_vector)
            if m > 0
        }
