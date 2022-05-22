import numpy as np
from scipy.optimize import nnls
from mixsol.components import Solution, Powder
from mixsol.helpers import name_to_components, components_to_name, calculate_molar_mass
from mixsol.digraph import DirectedGraph
import itertools as itt
import matplotlib.pyplot as plt
from random import sample


class Mixer:
    """class to calculate mixing paths from a set of stock solutions -> set of target solutions"""

    def __init__(self, stock_solutions: list, targets: dict = None):
        """Initialize the stock solutions + target solutions

        Args:
            stock_solutions (list): list of Solution objects defining starting solutions
            targets (dict, optional): dictionary of {Solution:volume} that define target solutions and volumes of each. Defaults to None.
        """
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
        ) = _solutions_to_matrix(self.solutions)
        self.stock_idx = [
            self.solutions.index(stock) for stock in stock_solutions
        ]  # rows of solution matrix that belong to stock solutions
        self.__solved = False

    def _is_plausible(
        self, solution_matrix: np.ndarray, target_vector: np.ndarray
    ) -> bool:
        """Check if the solution_matrix spans target vector values. If not, no chance of a successful mixture here!"""
        difference = solution_matrix - target_vector
        difference = difference.round(12)  # numerical solution has very tiny errors
        return not any([all(v > 0) or all(v < 0) for v in difference.T])

    def _calculate_mix(
        self,
        target: Solution,
        solution_indices: list = None,
        tolerance: float = 1e-4,
        min_fraction: float = 0,
    ) -> list:
        """Calculate mixture of solution (solution matrix rows) that combine to achieve the target solution


        Returns:
            list: list of float values defining the volume of each solution to mix to form target
        """
        if solution_indices is None:
            solution_indices = list(range(len(self.solutions)))
        A0 = self.solution_matrix[solution_indices]
        scale = A0[A0 > 0].min()  # smallest nonzero component
        A = (
            A0 / scale
        )  # change min value to 1, avoids rounding errors with trace components in Solution
        b = self._solution_to_vector(target) / scale
        if not self._is_plausible(A, b):
            return np.nan
        x, err = nnls(A.T, b, maxiter=1e3)
        x = x.round(12)  # numerical solution has very tiny errors
        if err > tolerance:
            return np.nan
        if np.logical_and(x > 0, x < min_fraction).any():
            return np.nan

        x_full = np.zeros((len(self.solutions),))
        for idx, x_ in zip(solution_indices, x):
            x_full[idx] = x_
        return x_full

    def _solution_to_vector(
        self,
        target: Solution,
        volume: float = 1,
        components: list = None,
    ) -> np.ndarray:
        """Convert a Solution object into a vector of molarities. vector matches the component ordering of the solution matrix

        Args:
            target (Solution): Target solution
            volume (float, optional): Volume of solution desired. Defaults to 1.
            components (list, optional): Subset of components to construct vector around. If None (default), uses component list for solution matrix.

        Returns:
            np.ndarray: nx1 vector of molarities in order of components list
        """
        # organize target solution into a matrix of total mols desired of each component
        if components is None:
            components = self.components
        target_matrix = np.zeros((len(components),))
        for m, c in enumerate(components):
            if c in target.solutes:
                target_matrix[m] = target.solutes[c] * volume
            elif c in target.solvent:
                target_matrix[m] = target.solvent[c] * volume
        return target_matrix.T

    def mix(
        self,
        target: Solution,
        volume: float,
        solution_indices: list = None,
        tolerance: float = 1e-5,
        min_volume: float = 0,
        verbose: bool = False,
        max_inputs: int = None,
        strategy: str = "least_inputs",
    ) -> np.ndarray:
        """Calculate mixture of stock solutions to achieve target solution

        Args:
            target (Solution): target solution
            volume (float): volume of target solution desired
            solution_indices (list, optional): list of solution (row) indices to consider for mixing. If None (default), assumes all solutions are valid.
            tolerance (float, optional): error threhold for target solution. Defaults to 1e-2.
            min_volume (float, optional): minimum volume that can be mixed from any single solution (useful for pipettes with a minimum aspiration volume). Defaults to 0.
            verbose (bool, optional): if True, returns all plausible mixture vectors. If False (default), only returns the best (largest minimum single volume transfer) vector.
            max_inputs (int, optional): maximum number of solutions to mix into the given target. Defaults to None (no limit).
            strategy (str, optional): strategy to select mixing inputs from the set of valid inputs.
                "least_inputs": select mixing inputs such that the smallest input volume is maximized. this should mix with the least number of input solutions
                "prefer_stock": select mixing inputs such that the number of non-stock inputs is minimized.

        Returns:
            np.ndarray: vector of solution volumes corresponding to rows in the solution matrix (self.solutions). If verbose=True, this will be a list of such vectors that all reach the target solution
        """
        if strategy not in ["least_inputs", "prefer_stock", "fastest"]:
            raise ValueError(
                "Mixing strategy must be 'least_inputs', 'prefer_stock', or 'fastest'"
            )
        min_fraction = min_volume / volume
        if solution_indices is None or solution_indices == "stock":
            solution_indices = self.stock_idx
        elif solution_indices == "all":
            solution_indices = list(range(len(self.solutions)))

        possible_mixtures = []
        if max_inputs is None:
            max_inputs = len(solution_indices) - 1
        max_inputs = min(
            volume // min_volume, max_inputs
        )  # min pipette volume may limit us to fewer inputs

        for i in range(1, max_inputs):
            for idx in itt.combinations(
                sample(solution_indices, len(solution_indices)), i
            ):
                x = self._calculate_mix(
                    target,
                    solution_indices=list(idx),
                    tolerance=tolerance,
                    min_fraction=min_fraction,
                )
                if not np.isnan(x).any():
                    possible_mixtures.append(
                        x.round(12) * volume
                    )  # remove floating pt precision errors
                    if strategy == "fastest":
                        break
            if len(possible_mixtures) > 0:
                break
        if len(possible_mixtures) == 0:
            return np.nan
        possible_mixtures = [pm for pm in np.unique(possible_mixtures, axis=0)]

        if strategy == "least_inputs":
            possible_mixtures.sort(
                key=lambda x: -min(x[x > 0])
            )  # sort such that the mixture with the largest minimum fraction is first
        elif strategy == "prefer_stock":
            nonstock_idx = [
                i for i in range(len(self.solutions)) if i not in self.stock_idx
            ]
            possible_mixtures.sort(
                key=lambda x: (np.sum(x[nonstock_idx] > 0).sum(), -min(x[x > 0]))
            )
        if verbose:
            return possible_mixtures

        return possible_mixtures[0]

    def _solve_adjacency_matrix(
        self,
        min_volume: float,
        max_inputs: int,
        max_generations: int,
        tolerance: float,
        strategy: str,
    ):
        """solves the mixing plan for all target solutions. Other target solutions can act as stepping stones to a target (multiple mixing generations).

        Args:
            min_volume (float): minimum volume for single liquid transfer. useful for pipettes with a minimum aspiration volume
            max_inputs (int): maximum number of solutions that can be mixed to achieve a target solution
            max_generations (int): maximum generations/rounds of mixing that are allowed
            strategy (str, optional): strategy to select mixing inputs from the set of valid inputs.
                "least_inputs": select mixing inputs such that the smallest input volume is maximized. this should mix with the least number of input solutions
                "prefer_stock": select mixing inputs such that the number of non-stock inputs is minimized.


        Returns:
            np.ndarray: adjacency matrix describing all volume transfers to achieve target solutions
        """
        self.__solved = False

        graph = np.zeros((len(self.solutions), len(self.solutions)))
        self.availableidx = self.stock_idx.copy()
        generation = 0
        n_remaining = len(self.solutions) - len(self.availableidx)
        n_remaining_lastiter = np.inf
        while n_remaining > 0:
            if generation > max_generations or n_remaining == n_remaining_lastiter:
                error_string = (
                    "Could not find a solution for the following solutions:\n"
                )
                for i, s in enumerate(self.solutions):
                    if i not in self.availableidx:
                        error_string += f"\t{s}\n"
                raise Exception(error_string)
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
                    tolerance=tolerance,
                    strategy=strategy,
                )
                if not np.isnan(x).any():
                    graph[i, :] = x
                    self.availableidx.append(i)
                    n_remaining -= 1
            generation += 1

        self.__solved = True
        return graph

    def solve(
        self,
        min_volume: float,
        max_inputs: int = None,
        max_generations: int = np.inf,
        tolerance: float = 1e-5,
        strategy: str = "least_inputs",
    ):
        """user-facing method to solve mixing strategy for all target solutions

        Args:
            min_volume (float): minimum volume for single liquid transfer. useful for pipettes with a minimum aspiration volume
            max_inputs (int): maximum number of solutions that can be mixed to achieve a target solution. Default is None.
            max_generations (int): maximum generations/rounds of mixing that are allowed. Default is np.inf (ie no limit)
            strategy (str, optional): strategy to select mixing inputs from the set of valid inputs.
                "least_inputs": select mixing inputs such that the smallest input volume is maximized. this should mix with the least number of input solutions
                "prefer_stock": select mixing inputs such that the number of non-stock inputs is minimized.

        """
        if max_inputs is None:
            max_inputs = len(self.solutions) - 1

        adjacency_matrix = self._solve_adjacency_matrix(
            min_volume=min_volume,
            max_inputs=max_inputs,
            max_generations=max_generations,
            tolerance=tolerance,
            strategy=strategy,
        )
        self.graph = DirectedGraph(adjacency_matrix)
        g_norm = self.graph._normalize(self.graph.g)
        v_end = np.array([self.target_volumes.get(soln, 0) for soln in self.solutions])
        v_needed = self.graph.propagate_load(v_end)
        self.initial_volumes_required = {
            solution: volume
            for solution, volume in zip(self.solutions, v_needed)
            if volume > 0
        }
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

    def all_possible_solutions(
        self,
        min_volume: float,
        target_volume: float,
        precision: float,
        max_inputs: int = 4,
    ):
        """Constructs a list of all Solution's that can be accessed from the set of stock solutions. Note that all generated solutions will have a molarity of one (doesn't change the actual contents, just the labeling, molarity is reflected in the solutes coefficients).

        Args:
            min_volume (float): minimum transfer volume
            target_volume (float): final volume of accessible Solution
            precision (float): increment of volume at which to mix stocks
            max_inputs (int, optional): Max number of stock solutinos that can be mixed. This may be overridden if min_volume/total_volume limits the number of inputs. Defaults to 4.

        Returns:
            list: list of accessible Solution's
        """
        max_inputs = min(
            max_inputs, int(np.floor(target_volume / min_volume))
        )  # possible that we are limited by volumes vs user guidance

        # ratios = self.__all_possible_ratios(
        #     num_inputs=max_inputs,
        #     min_fraction=min_volume / target_volume,
        #     steps=int(target_volume / precision),
        # )
        num_steps = int(target_volume / precision)
        ratios = []
        for c in itt.combinations_with_replacement(list(range(max_inputs)), num_steps):
            this_ratio = [c.count(i) / num_steps for i in range(max_inputs)]
            if min([r for r in this_ratio if r > 0]) < (min_volume / target_volume):
                continue
            ratios.append(this_ratio)

        solution_vectors = []
        for s in itt.combinations(self.solution_matrix, max_inputs):
            for r in ratios:
                row = np.array(s).T @ r
                solution_vectors.append(row)
        solution_vectors = np.array(solution_vectors).round(12)
        solution_vectors = np.unique(solution_vectors, axis=0)

        solutions = []
        for r in solution_vectors:
            solutes = {
                c: amt
                for idx, (c, amt) in enumerate(zip(self.components, r))
                if idx not in self.solvent_idx
            }
            solvents = {
                c: amt
                for idx, (c, amt) in enumerate(zip(self.components, r))
                if idx in self.solvent_idx
            }
            solutions.append(Solution(solutes=solutes, solvent=solvents, molarity=1))

        return solutions

    ### publishing methods
    def _check_if_solved(self):
        if not self.__solved:
            raise Exception("Solution mixing must be solved (using self.solve) first!")

    def print(self):
        """Prints the pipetting instructions in proper order to console in plain english. Will substitute solution names with their .alias if present"""
        self._check_if_solved()
        # for idx in self.stock_idx:
        #     volume_graph[idx, idx] = self.target_volumes.get(self.solutions[idx], 0)
        print("===== Stock Prep =====")
        for solution, volume in self.stock_volumes.items():
            if volume > 0:
                print(f"{volume:.2f} of {solution}")
        first = True
        for generation in self.transfers_per_generation:
            for source, transfers in generation.items():
                if not any([source != destination for destination in transfers]):
                    continue
                if first:
                    print(f"====== Mixing =====")
                    first = False
                print(f"Distribute {source}:")
                for destination, volume in transfers.items():
                    if destination != source:
                        print(f"\t{volume:.2f} to {destination}")

    def plot(self, ax=None):
        """Plots the pipetting instructions as a directed graph, layered by generation. Will substitute solution names with their .alias if present."""
        self._check_if_solved()
        nodes = {}
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        plt.sca(ax)
        # ax.set_aspect("equal")
        for i, gen in enumerate(self.mixing_order):
            xoff = 0.5 * (i % 2) + np.random.random() * 0.2
            for j, (node, sources) in enumerate(gen.items()):
                if node not in nodes:
                    x = j * 1.25 + xoff * len(self.mixing_order)
                    y = i + np.random.random() * 0.2 - 0.1
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
        plt.axis("off")
        plt.tight_layout()


class Weigher:
    """class to calculate weights of solute powders from a set of stock powders -> target solution of defined volume+molarity"""

    def __init__(self, powders: list):
        """Initialize the powder matrix

        Args:
            powders (list): List of Powder objects available for solution preparation.
        """
        self.powders = powders
        self.matrix, self.components = self._powders_to_matrix(powders)

    def _powders_to_matrix(self, powders: list) -> tuple:
        """Converts a list of powders into a matrix of mol/g of each component in each solution

        Args:
            powders (list): Powder objects

        Returns:
            solid_matrix (np.ndarray):  rows are powders, columns are components
            components (list): components that cover all powders (column key)


        """
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

    def _solution_to_vector(self, target: Solution, volume: float = 1) -> np.ndarray:
        """Convert a Solution object into a vector of molarities. vector matches the component ordering of the solution matrix

        Args:
            target (Solution): Target solution
            volume (float, optional): Volume of solution desired. Defaults to 1.
            components (list, optional): Subset of components to construct vector around. If None (default), uses component list for solution matrix.

        Returns:
            np.ndarray: nx1 vector of molarities in order of components list
        """
        # organize target solution into a matrix of total mols desired of each component
        target_matrix = np.zeros((len(self.components),))
        for m, component in enumerate(self.components):
            if component in target.solutes:
                target_matrix[m] = target.solutes[component] * volume
        return target_matrix.T

    def _filter_powders(self, target: Solution) -> list:
        """Filter out powders with components that are not present in the target solution (these powders will never be valid inputs).

        Args:
            target (Solution): target solution

        Returns:
            list: row indices of powders that are valid options for the target solution
        """
        idx_unusable = (
            set()
        )  # indices for which powders contain components that are not present in the target
        for component_idx in np.where(target == 0)[0]:
            present_in_powder = np.where(self.matrix[:, component_idx] > 0)[0]
            for powder_idx in present_in_powder:
                idx_unusable.add(powder_idx)
        idx_to_use = [i for i in range(self.matrix.shape[0]) if i not in idx_unusable]
        return idx_to_use

    def _lookup_powder(self, s: str):
        for idx, p in enumerate(self.powders):
            if p.alias == s or str(p) == s:
                return idx
        raise ValueError(f"Could not find powder {s} in this Weigher!")

    def get_weights(
        self, target: Solution, volume: float, tolerance: float = 1e-5
    ) -> dict:
        """calculate the weights of stock powders necessary to make a target Solution

        Args:
            target (Solution): target solution. molarity will be defined in the Solution object.
            volume (float): volume (L) of solution to make
            tolerance (float, optional): error in solution molarities to tolerate. This number is a bit arbitrary for now. Defaults to 1e-10.

        Raises:
            Exception: No plausible mix of weights to make this solution

        Returns:
            dict: dictionary of {powder:mass (g)} that should be dissolved to make the target solution.
        """
        target_vector = self._solution_to_vector(target=target, volume=volume)
        usable_powder_indices = self._filter_powders(target_vector)
        matrix = self.matrix[usable_powder_indices]
        mass_vector, error = nnls(matrix.T, target_vector.T, maxiter=1e4)
        mass_vector = mass_vector.round(12)
        if error > tolerance:  # TODO #1
            raise Exception(
                f"Could not achieve target solution from given powders. Error={error}, tolerance was {tolerance}"
            )
        return {
            str(self.powders[idx]): m
            for idx, m in zip(usable_powder_indices, mass_vector)
            if m > 0
        }

    def weights_to_solution(
        self, weights: dict, volume: float, solvent: str, molarity=None
    ) -> Solution:
        """Returns a Solution object generated by mixing the powder weights and solvent

        Args:
            weights (dict): {powder:mass(g)} dictionary. Powders must be present in this `Weigher`, and can be referred to by the `Powder.alias` or `Powder.formula` attributes
            volume (float): volume (L) of solvent powder is dispersed into
            solvent (str): Formula string for the solvent. ie "IPA", "DMF9_DMSO1", "EtOH0.5_IPA0.5".
            molarity (float, optional): Value to set molarity of resulting Solution. This can be a directly set float value, the name of a single component
                of the resulting `Solution` that molarity will be normalized to, or a list of names of components the sum of which will be used as the molarity.
                Note that this only affects the molarity value and string representation of the `Solution`, the actual `Solution` will be equivalent regardless of your choice here.
        Returns:
            Solution: solution object resulting from the powder mixing
        """
        v = np.zeros(len(self.components))
        for powder, mass in weights.items():
            m = self._lookup_powder(powder)
            v += self.matrix[m] * mass / volume  # v is in units of mol/L

        if isinstance(molarity, str):
            if molarity not in self.components:
                raise ValueError(
                    f"Cannot set molarity to {molarity} - this component does not exist in the resulting Solution!"
                )
            molarity = v[self.components.index(molarity)]
        elif isinstance(molarity, list):
            temp = 0
            for m in molarity:
                if m not in self.components:
                    raise ValueError(
                        f"Cannot set molarity to {molarity} - this component does not exist in the resulting Solution!"
                    )
                temp += v[self.components.index(m)]
            molarity = temp

        elif molarity is None:
            molarity = v.max()

        # v *= molarity
        solutes = {c: v for c, v in zip(self.components, v) if v > 0}
        return Solution(solutes=solutes, solvent=solvent, molarity=molarity)


def _order_components(solutions: list, components: list):
    """Order the components by their average order across the solutions. Solvents come after solutes"""
    component_rank = [[] for _ in components]
    for s in solutions:
        this_solutes = list(s.solutes.keys())
        this_solvent = list(s.solvent.keys())
        for c in s.solutes:
            component_rank[components.index(c)].append(this_solutes.index(c))
        for c in s.solvent:
            component_rank[components.index(c)].append(
                this_solvent.index(c) + 1000
            )  # penalize solvents so they come after solutes
    avg_component_rank = [np.mean(ranks) for ranks in component_rank]
    return [components[i] for i in np.argsort(avg_component_rank)]


def _solutions_to_matrix(solutions: list, components: list = None) -> tuple:
    """Converts a list of solutions into a matrix of molarity of each component in each solution

    Args:
        solutions (list): Solution objects
        components (list, optional): individual components that cover all solutions. If none, will generate this from the solution list

    Returns:
        solution_matrix (np.ndarray):  rows are solutions, columns are components
        solvent_idx (list): indices of columns that encode solvent components
        components (list): components that cover all solutions (column key)


    """
    if isinstance(solutions, Solution):
        solutions = [solutions]

    # get possible solution components from stock list
    if components is None:
        components = set()
        for s in solutions:
            components.update(s.solutes.keys(), s.solvent.keys())
        components = list(
            components
        )  # sets are not order-preserving, lists are - just safer this way

    components = _order_components(
        solutions, components
    )  # order components by average order across solutions, leads to more natural solution names

    # organize components into a stock matrix (rows = solutions, columns = components). keep track of which columns are solvents
    solution_matrix = np.zeros((len(solutions), len(components)))
    solvent_idx = set()
    for m, s in enumerate(solutions):
        for n, c in enumerate(components):
            if c in s.solutes:
                solution_matrix[m, n] = s.solutes[c]
            elif c in s.solvent:
                solution_matrix[m, n] = s.solvent[c]
                solvent_idx.add(n)
    solvent_idx = list(solvent_idx)

    return solution_matrix, solvent_idx, components


def interpolate(endpoints: list, divisor: int) -> list:
    """Generate a list of solutions that are a linear interpolation between the given solutions

    Args:
        endpoints (list): List of Solution objects to be interpolated between
        steps (int): number of steps to interpolate between endpoint solutions. 1 will return the original list, 2 will split into 50% increments, 3 split into 33% increments, etc.

    Returns:
        list: list of unique Solution objects resulting from the interpolation
    """
    if divisor <= 0:
        raise ValueError("Divisor must be greater than 0!")
    if not isinstance(divisor, int):
        raise ValueError("Divisor must be an integer!")

    solution_matrix, solvent_idx, components = _solutions_to_matrix(endpoints)
    solvent_components = [components[i] for i in solvent_idx]
    solution_idx = list(range(len(endpoints)))
    tweened_solutions = []
    for solution_indices in itt.combinations_with_replacement(solution_idx, divisor):
        svector = np.mean([solution_matrix[i] for i in solution_indices], axis=0)
        molarity = np.mean([endpoints[i].molarity for i in solution_indices])
        solutes = {
            c: v / molarity
            for c, v in zip(components, svector)
            if v > 0 and c not in solvent_components
        }
        solvent = {
            c: v
            for c, v in zip(components, svector)
            if v > 0 and c in solvent_components
        }

        new_solution = Solution(solutes=solutes, solvent=solvent, molarity=molarity)
        if new_solution not in tweened_solutions:
            tweened_solutions.append(new_solution)

    return tweened_solutions
