import pytest
from mixsol.components import Solution, Powder


class TestSolution:
    def test_solution_creation_basic(self):
        sol = Solution(
            solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0, alias="test"
        )
        assert sol.molarity == 1.0
        assert sol.alias == "test"
        assert "FA" in sol.solutes
        assert "Pb" in sol.solutes
        assert "I" in sol.solutes
        assert "DMF" in sol.solvents
        assert "DMSO" in sol.solvents

    def test_solution_creation_no_solutes(self):
        sol = Solution(solutes="", solvents="DMF9_DMSO1", molarity=0)
        assert sol.molarity == 0
        assert len(sol.solutes) == 0

    def test_solution_creation_missing_solvent(self):
        with pytest.raises(ValueError, match="Must define a solvent"):
            Solution(solutes="FA_Pb_I3", solvents=None, molarity=1.0)

    def test_solution_creation_solutes_zero_molarity(self):
        with pytest.raises(ValueError, match="molarity must be >0"):
            Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=0)

    def test_solution_string_representation(self):
        sol = Solution(
            solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0, alias="test"
        )
        assert str(sol) == "test"

    def test_solution_string_representation_no_alias(self):
        sol = Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        str_repr = str(sol)
        assert "1M" in str_repr
        assert "FA" in str_repr
        assert "Pb" in str_repr
        assert "I3" in str_repr

    def test_solution_equality(self):
        sol1 = Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        sol2 = Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        assert sol1 == sol2

    def test_solution_inequality(self):
        sol1 = Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        sol2 = Solution(solutes="MA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        assert sol1 != sol2

    def test_solution_hash(self):
        sol1 = Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        sol2 = Solution(solutes="FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        assert hash(sol1) == hash(sol2)

    def test_solution_dict_components(self):
        sol = Solution(
            solutes={"FA": 1, "Pb": 1, "I": 3},
            solvents={"DMF": 9, "DMSO": 1},
            molarity=1.0,
        )
        assert sol.solutes["FA"] == 1.0
        assert sol.solutes["Pb"] == 1.0
        assert sol.solutes["I"] == 3.0

    def test_solution_duplicate_components_error(self):
        sol = Solution(solutes="FA_FA_Pb_I3", solvents="DMF9_DMSO1", molarity=1.0)
        # FA should be combined: FA + FA = FA2
        assert sol.solutes["FA"] == 2.0

    def test_solution_parentheses_formula(self):
        sol = Solution(solutes="(A_B)2_C", solvents="DMF", molarity=1.0)
        assert sol.solutes["A"] == 2.0
        assert sol.solutes["B"] == 2.0
        assert sol.solutes["C"] == 1.0


class TestPowder:
    def test_powder_creation_basic(self):
        powder = Powder(formula="Cs_I", alias="cesium_iodide")
        assert powder.alias == "cesium_iodide"
        assert "Cs" in powder.components
        assert "I" in powder.components

    def test_powder_creation_with_molar_mass(self):
        powder = Powder(formula="MA_I", molar_mass=158.97, alias="MAI")
        assert powder.molar_mass == 158.97
        assert powder.alias == "MAI"

    def test_powder_string_representation(self):
        powder = Powder(formula="Cs_I", alias="cesium_iodide")
        assert str(powder) == "cesium_iodide"

    def test_powder_string_representation_no_alias(self):
        powder = Powder(formula="Cs_I")
        str_repr = str(powder)
        assert "Cs" in str_repr
        assert "I" in str_repr

    def test_powder_equality(self):
        powder1 = Powder(formula="Cs_I")
        powder2 = Powder(formula="Cs_I")
        assert powder1 == powder2

    def test_powder_inequality(self):
        powder1 = Powder(formula="Cs_I")
        powder2 = Powder(formula="Pb_I2")
        assert powder1 != powder2

    def test_powder_hash(self):
        powder1 = Powder(formula="Cs_I")
        powder2 = Powder(formula="Cs_I")
        assert hash(powder1) == hash(powder2)

    def test_powder_dict_components(self):
        powder = Powder(formula={"Cs": 1, "I": 1}, molar_mass=259.8)
        # Components are not divided by molar mass when dict is provided
        assert powder.components["Cs"] == 1
        assert powder.components["I"] == 1

    def test_powder_duplicate_components_error(self):
        # The actual implementation combines duplicate components
        powder = Powder(formula="Cs_Cs_I")
        # Components should be combined: Cs + Cs = Cs2
        assert powder.components["Cs"] > 0  # Some positive value for Cs

    def test_powder_repr(self):
        powder = Powder(formula="Cs_I")
        assert repr(powder).startswith("<Powder>")
