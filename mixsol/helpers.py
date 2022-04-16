import numpy as np
from molmass import Formula
import re

## parsing formulae -> dictionaries
def components_to_name(
    components: dict, delimiter: str = "_", factor: float = 1
) -> str:
    """Convert a dictionary of components into a string

    Args:
        components (dict): {component(str):amount(float)}
        delimiter (str, optional): will be inserted between each component. Defaults to "_".
        factor (float, optional): factor to multiply all amounts by. Defaults to 1

    Returns:
        str: string of (component)(amount)(delimiter) repeat units
    """
    composition_label = ""
    for c, nraw in components.items():
        n = nraw * factor
        if n == 0:
            continue
        elif n == 1:
            composition_label += "{0}{2}".format(c, n, delimiter)
        else:
            composition_label += "{0}{1:.3g}{2}".format(c, n, delimiter)

    return composition_label[:-1]


def __digest_string(s, factor=1, delimiter="_"):
    components = {}
    groups = re.findall(f"{delimiter}?(\D*)(\D*[+-]?[0-9]*[.]?[0-9]+){delimiter}?", s)
    parenthesized = 0
    for species, amount in groups:
        if "(" in species:
            parenthesized += 1
        if parenthesized == 0:
            components[species] = float(amount) * factor
        if ")" in species:
            parenthesized -= 1
    return components


def name_to_components(
    name: str, factor: float = 1, delimiter: str = "_", components=None
) -> dict:
    """
    given a chemical formula, returns dictionary with individual components/amounts
    expected name format = 'MA0.5_FA0.5_Pb1_I2_Br1'.
    would return dictionary with keys ['MA, FA', 'Pb', 'I', 'Br'] and values [0.5,.05,1,2,1]*factor

    Args:
        name (str): formula string
        factor (float): factor to multiply all amount values in the string by. Defaults to 1.
        delimiter (str): indicator string to split the formula into components. Defaults to "_"
        components: should be left at None, used for internal function recursion!
    Returns:
        dict: {component:amount}
    """
    if components is None:
        components = {}

    if len(name) == 0:
        return {}

    name_ = name
    delimiter_indices = [i for i, letter in enumerate(name) if letter == delimiter]
    for idx in delimiter_indices[::-1]:
        if name[idx - 1].isalpha():
            name_ = name_[:idx] + "1" + name_[idx:]
    if name_[-1].isalpha():
        name_ = name_ + "1"

    for comp, amt in __digest_string(name_, factor=factor, delimiter=delimiter).items():
        if amt == 0:
            continue
        if comp in components:
            components[comp] += amt
        else:
            components[comp] = amt
    parenthesized = re.findall("\((.*)\)(\D*[+-]?[0-9]*[.]?[0-9]+)", name_)
    # parenthesized = re.findall("\(([^\)]*)\)(\D*[0-9]*[.]?[0-9]+)", name_)
    for group, group_factor in parenthesized:
        components = name_to_components(
            name=group,
            delimiter=delimiter,
            factor=factor * float(group_factor),
            components=components,
        )

    return components


## getting molar mass - thanks @ molmass!
def calculate_molar_mass(formula, delimiter="_") -> float:
    """Given a formula string, try to get the molar mass using the molmass package

    Args:
        formula (str/dict): chemical formula to get molar mass for. Can also be a dictionary of {element:amount}
        delimiter (str, optional): delimiter character/string to remove from formula, since molmass does not expect a delimiter. Defaults to "_".

    Raises:
        ValueError: molmass could not return a molar mass. Often this is because the formula contains non-elemental units (ie MA for methylammonium, which is actually C,N,and H's)

    Returns:
        float: molar mass (g/mol)
    """
    if isinstance(formula, dict):
        fstr = ""
        for el, amt in formula.items():
            if amt == 0:
                continue
            elif amt == 1:
                fstr += f"{el}{delimiter}"
            else:
                fstr += f"{el}{amt}{delimiter}"
        formula = fstr[:-1]  # remove trailing delimiter
    try:
        return Formula(formula.replace(delimiter, "")).mass
    except:
        raise ValueError(
            f"Could not guess the molar mass for formula {formula}. Maybe there are non-elemental formula units?\n Either replace all formula units with elemental components, or manually input using the molar_mass argument."
        )
