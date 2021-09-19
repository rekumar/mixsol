import numpy as np
from molmass import Formula

#### name parsing helper functions
def components_to_name(components, delimiter="_"):
    composition_label = ""
    for c, n in components.items():
        if n == 0:
            continue
        elif n == 1:
            composition_label += "{0}{2}".format(c, n, delimiter)
        else:
            composition_label += "{0}{1:.3g}{2}".format(c, n, delimiter)

    return composition_label[:-1]


def name_to_components(
    name,
    factor=1,
    delimiter="_",
):
    """
    given a chemical formula, returns dictionary with individual components/amounts
    expected name format = 'MA0.5_FA0.5_Pb1_I2_Br1'.
    would return dictionary with keys ['MA, FA', 'Pb', 'I', 'Br'] and values [0.5,.05,1,2,1]*factor
    """
    components = {}
    for part in name.split(delimiter):
        species = part
        count = 1.0
        for l in range(len(part), 0, -1):
            try:
                count = float(part[-l:])
                species = part[:-l]
                break
            except:
                pass
        if species == "":
            continue
        components[species] = count * factor
    return components


def solutions_to_matrix(solutions: list):
    if isinstance(solutions, Solution):
        solutions = [solutions]

    # get possible solution components from stock list
    components = set()
    for s in solutions:
        components.update(s.solute_dict.keys(), s.solvent_dict.keys())
    components = list(
        components
    )  # sets are not order-preserving, lists are - just safer this way

    # organize components into a stock matrix, keep track of which rows are solvents
    stock_matrix = np.zeros((len(components), len(solutions)))
    solvent_idx = set()
    for n, s in enumerate(solutions):
        for m, c in enumerate(components):
            if c in s.solute_dict:
                stock_matrix[m, n] = s.solute_dict[c] * s.molarity
            elif c in s.solvent_dict:
                stock_matrix[m, n] = s.solvent_dict[c]
                solvent_idx.add(m)
    solvent_idx = list(solvent_idx)

    return stock_matrix


def calculate_molar_mass(formula, delimiter="_"):
    try:
        return Formula(formula.replace(delimiter, "")).mass
    except:
        raise ValueError(
            f"Could not guess the molar mass for formula {formula}. Maybe there are non-elemental formula units?\n Either replace all formula units with elemental components, or manually input using the molar_mass argument."
        )


class Solution:
    def __init__(
        self, solvent: str, solutes: str = "", molarity: float = 0, alias: str = None
    ):
        if solutes != "" and molarity == 0:
            raise ValueError(
                "If the solution contains solutes, the molarity must be >0!"
            )
        if solutes == "":
            molarity = 1

        self.molarity = molarity
        self.solute_dict = name_to_components(solutes, factor=molarity, delimiter="_")
        self.solutes = components_to_name(self.solute_dict, delimiter="_")
        self.solvent_dict = name_to_components(solvent, factor=1, delimiter="_")
        self.solvent = components_to_name(self.solvent_dict, delimiter="_")
        total_solvent_amt = sum(self.solvent_dict.values())
        self.solvent_dict = {
            k: v / total_solvent_amt for k, v in self.solvent_dict.items()
        }  # normalize so total solvent amount is 1.0

        self.alias = alias

    def to_dict(self):
        out = {
            "solutes": self.solutes,
            "molarity": self.molarity,
            "solvent": self.solvent,
            "well": self.well,
        }
        return out

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        if self.alias is not None:
            return self.alias
        if self.solutes == "":  # no solutes, just a solvent
            return f"{self.solvent}"
        return f"{round(self.molarity,2)}M {self.solutes} in {self.solvent}"

    def __repr__(self):
        return f"<Solution>" + str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.solutes == other.solutes
                and self.molarity == other.molarity
                and self.solvent == other.solvent
            )
        else:
            return False

    def __key(self):
        return (self.solutes, self.molarity, self.solvent)

    def __hash__(self):
        return hash(self.__key())


class Powder:
    def __init__(self, formula: str, molar_mass: float = None, alias: str = None):
        if molar_mass is None:
            self.molar_mass = calculate_molar_mass(formula, "_")
        else:
            self.molar_mass = molar_mass
        self.formula = formula
        self.components = name_to_components(
            formula, factor=1 / self.molar_mass, delimiter="_"
        )
        self.alias = alias

    def __str__(self):
        if self.alias is not None:
            return self.alias
        else:
            return self.formula

    def __repr__(self):
        return f"<Powder>" + str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.formula == other.formula
        else:
            return False

    def __key(self):
        return (self.formula, self.molar_mass)

    def __hash__(self):
        return hash(self.__key())
