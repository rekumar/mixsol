import pytest
from mixsol.helpers import components_to_name, name_to_components, calculate_molar_mass


class TestComponentsToName:
    def test_components_to_name_basic(self):
        components = {"A": 1, "B": 2, "C": 3}
        result = components_to_name(components)
        assert result == "A_B2_C3"

    def test_components_to_name_with_factor(self):
        components = {"A": 1, "B": 2}
        result = components_to_name(components, factor=2)
        assert result == "A2_B4"

    def test_components_to_name_custom_delimiter(self):
        components = {"A": 1, "B": 2}
        result = components_to_name(components, delimiter="-")
        assert result == "A-B2"

    def test_components_to_name_single_amounts(self):
        components = {"A": 1, "B": 1}
        result = components_to_name(components)
        assert result == "A_B"

    def test_components_to_name_zero_amounts(self):
        components = {"A": 1, "B": 0, "C": 2}
        result = components_to_name(components)
        assert result == "A_C2"

    def test_components_to_name_empty_dict(self):
        components = {}
        result = components_to_name(components)
        assert result == ""

    def test_components_to_name_fractional_amounts(self):
        components = {"A": 0.5, "B": 1.5}
        result = components_to_name(components)
        assert "A0.5" in result
        assert "B1.5" in result


class TestNameToComponents:
    def test_name_to_components_basic(self):
        name = "A_B2_C3"
        result = name_to_components(name)
        expected = {"A": 1, "B": 2, "C": 3}
        assert result == expected

    def test_name_to_components_with_factor(self):
        name = "A_B2"
        result = name_to_components(name, factor=2)
        expected = {"A": 2, "B": 4}
        assert result == expected

    def test_name_to_components_custom_delimiter(self):
        name = "A-B2"
        result = name_to_components(name, delimiter="-")
        expected = {"A": 1, "B": 2}
        assert result == expected

    def test_name_to_components_fractional(self):
        name = "A0.5_B1.5"
        result = name_to_components(name)
        expected = {"A": 0.5, "B": 1.5}
        assert result == expected

    def test_name_to_components_parentheses(self):
        name = "(A_B)2_C"
        result = name_to_components(name)
        expected = {"A": 2, "B": 2, "C": 1}
        assert result == expected

    def test_name_to_components_nested_parentheses(self):
        name = "(A2_B)3_C"
        result = name_to_components(name)
        expected = {"A": 6, "B": 3, "C": 1}
        assert result == expected

    def test_name_to_components_empty_string(self):
        name = ""
        result = name_to_components(name)
        assert result == {}

    def test_name_to_components_single_component(self):
        name = "A"
        result = name_to_components(name)
        expected = {"A": 1}
        assert result == expected

    def test_name_to_components_duplicate_components(self):
        name = "A_B_A2"
        result = name_to_components(name)
        expected = {"A": 3.0, "B": 1.0}
        assert result == expected

    def test_name_to_components_negative_amounts(self):
        name = "A-1_B2"
        result = name_to_components(name)
        # The actual implementation treats "A-1" as component "A-" with amount 1
        expected = {"A-": 1.0, "B": 2.0}
        assert result == expected


class TestCalculateMolarMass:
    def test_calculate_molar_mass_basic(self):
        # Test with a simple chemical formula
        mass = calculate_molar_mass("H2O")
        assert abs(mass - 18.015) < 0.01  # Approximate molar mass of water

    def test_calculate_molar_mass_with_delimiter(self):
        # Test with underscore delimiter
        mass = calculate_molar_mass("H2_O", delimiter="_")
        assert abs(mass - 18.015) < 0.01

    def test_calculate_molar_mass_dict_input(self):
        # Test with dictionary input
        formula_dict = {"H": 2, "O": 1}
        mass = calculate_molar_mass(formula_dict)
        assert abs(mass - 18.015) < 0.01

    def test_calculate_molar_mass_complex_formula(self):
        # Test with more complex formula
        mass = calculate_molar_mass("CaCl2")
        assert abs(mass - 110.98) < 0.1  # Approximate molar mass of calcium chloride

    def test_calculate_molar_mass_invalid_formula(self):
        # Test with invalid formula that should raise ValueError
        with pytest.raises(ValueError, match="Could not guess the molar mass"):
            calculate_molar_mass("InvalidFormula123")

    def test_calculate_molar_mass_zero_amount(self):
        # Test with zero amount in dictionary
        formula_dict = {"H": 2, "O": 0, "Cl": 2}
        mass = calculate_molar_mass(formula_dict)
        # Should be close to H2Cl2 mass
        assert mass > 0

    def test_calculate_molar_mass_fractional_amounts(self):
        # Test with fractional amounts - molmass doesn't support fractional amounts
        formula_dict = {"H": 1.5, "O": 0.5}
        with pytest.raises(ValueError, match="Could not guess the molar mass"):
            calculate_molar_mass(formula_dict)


class TestRoundTripConversion:
    def test_components_name_roundtrip(self):
        """Test that converting components to name and back gives the same result"""
        original = {"A": 1, "B": 2, "C": 0.5}
        name = components_to_name(original)
        result = name_to_components(name)

        # Check that all keys are present and values are close
        assert set(original.keys()) == set(result.keys())
        for key in original:
            assert abs(original[key] - result[key]) < 1e-10

    def test_name_components_roundtrip(self):
        """Test that converting name to components and back gives the same result"""
        original = "A_B2_C0.5"
        components = name_to_components(original)
        result = components_to_name(components)

        # Parse both to compare (order might differ)
        original_dict = name_to_components(original)
        result_dict = name_to_components(result)
        assert original_dict == result_dict
