from mixsol.helpers import components_to_name, name_to_components, calculate_molar_mass
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

        self.molarity = molarity
        self.alias = alias

        self.solutes = self.__digest_components(solutes, factor=self.molarity)
        if len(self.solutes) == 0:
            molarity = 1
        else:
            if molarity <= 0:
                raise ValueError(
                    "If the solution contains solutes, the molarity must be >0!"
                )
        total_solute_amt = sum(self.solutes.values())
        self._solute_dict_norm = {
            k: v / total_solute_amt for k, v in self.solutes.items()
        }  # normalize so total solute amount is 1.0. used for hashing/comparison to other Solution's
        self.__solute_str_norm = json.dumps(
            {k: round(v, 5) for k, v in self._solute_dict_norm.items()}, sort_keys=True
        )  # used for hashing

        self.solvent = self.__digest_components(solvent, factor=1)
        total_solvent_amt = sum(self.solvent.values())
        self.solvent = {
            k: v / total_solvent_amt for k, v in self.solvent.items()
        }  # normalize so total solvent amount is 1.0
        self.__solvent_str_norm = json.dumps(
            {k: round(v, 5) for k, v in self.solvent.items()}, sort_keys=True
        )  # used for hashing

    def __digest_components(self, components, factor):
        if isinstance(components, str):
            components = name_to_components(components, factor=factor)
        elif isinstance(components, dict):
            components = {k: v * factor for k, v in components.items()}
        else:
            raise ValueError(
                "Components must be given as an underscore-delimited string (eg Cs_Pb_I3) or dictionary (eg {'Cs':1, 'Pb':1, 'I':3'})!"
            )
        if len(components) != len(set(components.keys())):
            raise ValueError(
                "All components of solutes/solvents must be unique - did you repeat an element/molecule component?"
            )
        components = {k: round(v, 12) for k, v in components.items()}
        return components

    def __str__(self):
        if self.alias is not None:
            return self.alias
        if len(self.solutes) == 0:  # no solutes, just a solvent
            return components_to_name(self.solvent)
        return f"{self.molarity:.2g}M {components_to_name(self.solutes, factor=1/self.molarity)} in {components_to_name(self.solvent)}"

    def __repr__(self):
        return f"<Solution> " + str(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for d1, d2 in zip(
            [self.solutes, self.solvent],
            [other.solutes, other.solvent],
        ):
            if d1.keys() != d2.keys():
                return False
            for k in d1.keys():
                if d2[k] == 0:  # catch for divide by zero issue
                    if d1[k] != 0:
                        return False
                elif (
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
            self.molar_mass = calculate_molar_mass(formula, delimiter="_")
        else:
            self.molar_mass = molar_mass
        self.components = self.__digest_components(formula, factor=1 / self.molar_mass)
        self.formula = components_to_name(self.components, factor=self.molar_mass)
        self.alias = alias

    def __digest_components(self, components, factor):
        if isinstance(components, str):
            components = name_to_components(components, factor=factor)
        elif isinstance(components, dict):
            pass
        else:
            raise ValueError(
                "Components must be given as an underscore-delimited string (eg Cs_Pb_I3) or dictionary (eg {'Cs':1, 'Pb':1, 'I':3'})!"
            )
        if len(components) != len(set(components.keys())):
            raise ValueError(
                "All components of the formula must be unique - did you repeat an element/molecule component?"
            )
        return components

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
