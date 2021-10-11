from mixsol.helpers import name_to_components, calculate_molar_mass
import numpy as np
import json


class Solution:
    def __init__(
        self,
        solutes: str = "",
        solvent: str = None,
        molarity: float = 0,
        alias: str = None,
    ):
        if solvent is None:
            raise ValueError("Must define a solvent!")
        if solutes != "" and molarity == 0:
            raise ValueError(
                "If the solution contains solutes, the molarity must be >0!"
            )
        if solutes == "":
            molarity = 1

        self.molarity = molarity
        self.alias = alias

        self.solutes = solutes
        self.solute_dict = name_to_components(
            solutes, factor=molarity
        )  # dictionary of molarity of each component
        total_solute_amt = sum(self.solute_dict.values())
        self._solute_dict_norm = {
            k: v / total_solute_amt for k, v in self.solute_dict.items()
        }  # normalize so total solute amount is 1.0. used for hashing/comparison to other Solution's
        self.__solute_str_norm = json.dumps(
            {k: round(v, 5) for k, v in self._solute_dict_norm.items()}, sort_keys=True
        )  # used for hashing

        self.solvent = solvent
        self.solvent_dict = name_to_components(solvent)
        # self.solvent = components_to_name(self.solvent_dict, delimiter="_")
        total_solvent_amt = sum(self.solvent_dict.values())
        self.solvent_dict = {
            k: v / total_solvent_amt for k, v in self.solvent_dict.items()
        }  # normalize so total solvent amount is 1.0
        self.__solvent_str_norm = json.dumps(
            {k: round(v, 5) for k, v in self.solvent_dict.items()}, sort_keys=True
        )  # used for hashing

    def __str__(self):
        if self.alias is not None:
            return self.alias
        if self.solutes == "":  # no solutes, just a solvent
            return f"{self.solvent}"
        return f"{round(self.molarity,2)}M {self.solutes} in {self.solvent}"

    def __repr__(self):
        return f"<Solution> " + str(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for d1, d2 in zip(
            [self.solute_dict, self.solvent_dict],
            [other.solute_dict, other.solvent_dict],
        ):
            if d1.keys() != d2.keys():
                return False
            for k in d1.keys():
                if (
                    np.abs(1 - (d1[k] / d2[k])) > 0.0001
                ):  # tolerance to accomodate rounding errors
                    return False
        return True

    def __key(self):
        return (self.__solvent_str_norm, self.__solute_str_norm)

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
