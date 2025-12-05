import pytest
import Sustainable_Code_Demo.definitions as defs

class Test_Devision:
    a = 10
    def test_basics(self):
        b = 5
        assert defs.division(self.a, b) == 2

    def test_zero_div(self):
        b = 0
        with pytest.raises(ZeroDivisionError):
            defs.division(self.a, b)