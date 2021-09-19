#### name parsing helper functions
def components_to_name(components, delimiter="_"):
    composition_label = ""
    for c, n in components.items():
        if n > 0:
            composition_label += "{0}{1:.2f}{2}".format(c, n, delimiter)

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
