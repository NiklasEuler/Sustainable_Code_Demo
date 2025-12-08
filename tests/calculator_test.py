import pytest
import Sustainable_Code_Demo.calculator as defs

class Test_Addition:
    a = 10
    def test_basics(self):
        b = 5
        assert defs.addition(self.a, b) == 15

    def test_negative(self):
        b = -3
        assert defs.addition(self.a, b) == 7

class Test_Subtraction:
    a = 10
    def test_basics(self):
        b = 5
        assert defs.subtraction(self.a, b) == 5

    def test_negative(self):
        b = -3
        assert defs.subtraction(self.a, b) == 13

class Test_Multiplication:
    a = 10
    def test_basics(self):
        b = 5
        assert defs.multiplication(self.a, b) == 50

    def test_negative(self):
        b = -3
        assert defs.multiplication(self.a, b) == -30

class Test_Devision:
    a = 10
    def test_basics(self):
        b = 5
        assert defs.division(self.a, b) == 2

    def test_zero_div(self):
        b = 0
        with pytest.raises(ZeroDivisionError):
            defs.division(self.a, b)