from mixsol.helpers import components_to_name, name_to_components, calculate_molar_mass
import json


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
        self.solutes = solutes
        self.solute_dict = name_to_components(solutes, delimiter="_", factor=1)
        total_solute_amt = sum(self.solute_dict.values())
        solute_dict_norm = {
            k: v / total_solute_amt for k, v in self.solute_dict.items()
        }  # normalize so total solute amount is 1.0. used for hashing/comparison to other Solution's
        self.__solute_str_norm = json.dumps(solute_dict_norm, sort_keys=True)
        self.solvent = solvent
        self.solvent_dict = name_to_components(solvent, factor=1, delimiter="_")
        # self.solvent = components_to_name(self.solvent_dict, delimiter="_")
        total_solvent_amt = sum(self.solvent_dict.values())
        self.solvent_dict = {
            k: v / total_solvent_amt for k, v in self.solvent_dict.items()
        }  # normalize so total solvent amount is 1.0
        self.__solvent_str_norm = json.dumps(self.solvent_dict, sort_keys=True)
        self.alias = alias

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
                self.solute_dict == other.solute_dict
                and self.molarity == other.molarity
                and self.solvent_dict == other.solvent_dict
            )
        else:
            return False

    def __key(self):
        return (self.__solvent_str_norm, self.molarity, self.__solute_str_norm)

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
