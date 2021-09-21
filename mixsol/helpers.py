import numpy as np
from molmass import Formula


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


def name_to_components(
    name: str,
    factor: float = 1,
    delimiter: str = "_",
) -> dict:
    """
    given a chemical formula, returns dictionary with individual components/amounts
    expected name format = 'MA0.5_FA0.5_Pb1_I2_Br1'.
    would return dictionary with keys ['MA, FA', 'Pb', 'I', 'Br'] and values [0.5,.05,1,2,1]*factor

    Args:
        name (str): formula string
        factor (float): factor to multiply all amount values in the string by. Defaults to 1.
        delimiter (str): indicator string to split the formula into components. Defaults to "_"

    Returns:
        dict: {component:amount}
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


## getting molar mass - thanks @ molmass!
def calculate_molar_mass(formula, delimiter="_") -> float:
    """Given a formula string, try to get the molar mass using the molmass package

    Args:
        formula (str): chemical formula to get molar mass for
        delimiter (str, optional): delimiter character/string to remove from formula, since molmass does not expect a delimiter. Defaults to "_".

    Raises:
        ValueError: molmass could not return a molar mass. Often this is because the formula contains non-elemental units (ie MA for methylammonium, which is actually C,N,and H's)

    Returns:
        float: molar mass (g/mol)
    """
    try:
        return Formula(formula.replace(delimiter, "")).mass
    except:
        raise ValueError(
            f"Could not guess the molar mass for formula {formula}. Maybe there are non-elemental formula units?\n Either replace all formula units with elemental components, or manually input using the molar_mass argument."
        )
