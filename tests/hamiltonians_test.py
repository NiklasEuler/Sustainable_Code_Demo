import pytest
from scipy.sparse import kron
import Sustainable_Code_Demo.hamiltonians as hams

class Test_Spin_Half_Operators:

    def test_trace_sx(self):
        trace_sx = hams.trace(hams.sx)
        assert trace_sx == 0

    def test_trace_sy(self):
        trace_sy = hams.trace(hams.sy)
        assert trace_sy == 0

    def test_trace_sz(self):
        trace_sz = hams.trace(hams.sz)
        assert trace_sz == 0

    def test_trace_I(self):
        trace_I = hams.trace(hams.I)
        assert trace_I == 2

    def test_square_sx(self):
        sx_squared = hams.sx @ hams.sx
        expected = 0.25 * hams.I
        assert (sx_squared != expected).nnz == 0

    def test_square_sy(self):
        sy_squared = hams.sy @ hams.sy
        expected = 0.25 * hams.I
        assert (sy_squared != expected).nnz == 0

    def test_square_sz(self):
        sz_squared = hams.sz @ hams.sz
        expected = 0.25 * hams.I
        assert (sz_squared != expected).nnz == 0

class Test_One_Spin_Operator:

    L = 3
    spin_op = hams.sy

    def test_one_spin_op_site0(self):
        idx = 0
        op = hams.one_spin_op(self.L, idx, self.spin_op)
        expected = kron(self.spin_op, kron(hams.I, hams.I))
        assert (op != expected).nnz == 0

    def test_one_spin_op_site1(self):
        idx = 1
        op = hams.one_spin_op(self.L, idx, self.spin_op)
        expected = kron(hams.I, kron(self.spin_op, hams.I))
        assert (op != expected).nnz == 0

    def test_one_spin_op_site2(self):
        idx = 2
        op = hams.one_spin_op(self.L, idx, self.spin_op)
        expected = kron(hams.I, kron(hams.I, self.spin_op))
        assert (op != expected).nnz == 0

class Test_Two_Spin_Operator:
    
    L = 5
    spin_op1 = hams.sx
    spin_op2 = hams.sy

    def test_two_spin_op_sites01(self):
        idx1 = 0
        idx2 = 1
        op = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        expected = kron(self.spin_op1, kron(self.spin_op2, kron(hams.I, kron(hams.I, hams.I))))
        assert (op != expected).nnz == 0

    def test_two_spin_op_sites24(self):
        idx1 = 2
        idx2 = 4
        op = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        expected = kron(hams.I, kron(hams.I, kron(self.spin_op1, kron(hams.I, self.spin_op2))))
        assert (op != expected).nnz == 0

    def test_two_spin_op_sites13(self):
        idx1 = 1
        idx2 = 3
        op = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        expected = kron(hams.I, kron(self.spin_op1, kron(hams.I, kron(self.spin_op2, hams.I))))
        assert (op != expected).nnz == 0

    def test_two_spin_op_sites_order(self):
        idx1 = 3
        idx2 = 1
        order_1 = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        order_2 = hams.two_spin_op(self.L, idx2, idx1, self.spin_op2, self.spin_op1)
        assert (order_1 != order_2).nnz == 0

    def test_two_spin_op_same_site(self):
        idx1 = 2
        idx2 = 2
        with pytest.raises(ValueError):
            hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
            