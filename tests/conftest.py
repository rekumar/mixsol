import pytest
from mixsol.components import Solution, Powder


@pytest.fixture
def basic_solution():
    """A basic solution for testing"""
    return Solution(solutes="FA_Pb_I3", solvents="DMF", molarity=1.0, alias="basic_sol")


@pytest.fixture
def basic_powder():
    """A basic powder for testing"""
    return Powder(formula="Cs_I", alias="cesium_iodide")


@pytest.fixture
def stock_solutions():
    """A list of stock solutions for mixer testing"""
    return [
        Solution(solutes="FA_Pb_I3", solvents="DMF", molarity=1.0, alias="FAPI"),
        Solution(solutes="MA_Pb_I3", solvents="DMF", molarity=1.0, alias="MAPI"),
        Solution(solutes="Cs_Pb_I3", solvents="DMF", molarity=1.0, alias="CsPI"),
    ]


@pytest.fixture
def basic_powders():
    """A list of basic powders for weigher testing"""
    return [
        Powder(formula="Cs_I", molar_mass=259.8, alias="CsI"),
        Powder(formula="Pb_I2", molar_mass=461.0, alias="PbI2"),
        Powder(formula="Pb_Br2", molar_mass=367.0, alias="PbBr2"),
    ]
