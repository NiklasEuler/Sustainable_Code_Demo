import pytest
import numpy as np
from scipy.sparse import kron, csr_matrix
import Sustainable_Code_Demo.hamiltonians as hams

class Test_Spin_Half_Operators:

    def test_sx_dense(self):
        expected = 0.5 * np.array([[0, 1], [1, 0]])
        assert np.allclose(hams.sx.todense(), expected)

    def test_sy_dense(self):
        expected = 0.5 * np.array([[0, -1j], [1j, 0]])
        assert np.allclose(hams.sy.todense(), expected)

    def test_sz_dense(self):
        expected = 0.5 * np.array([[1, 0], [0, -1]])
        assert np.allclose(hams.sz.todense(), expected)
    
    def test_I_dense(self):
        expected = np.array([[1, 0], [0, 1]])
        assert np.allclose(hams.I.todense(), expected)

    def test_trace_sx(self):
        trace_sx = hams.trace(hams.sx)
        assert trace_sx == pytest.approx(0)

    def test_trace_sy(self):
        trace_sy = hams.trace(hams.sy)
        assert trace_sy == pytest.approx(0)

    def test_trace_sz(self):
        trace_sz = hams.trace(hams.sz)
        assert trace_sz == pytest.approx(0)

    def test_trace_I(self):
        trace_I = hams.trace(hams.I)
        assert trace_I == pytest.approx(2)

    def test_square_sx(self):
        sx_squared = hams.sx @ hams.sx
        expected = 0.25 * hams.I
        assert np.allclose(sx_squared.todense(), expected.todense())

    def test_square_sy(self):
        sy_squared = hams.sy @ hams.sy
        expected = 0.25 * hams.I
        assert np.allclose(sy_squared.todense(), expected.todense())

    def test_square_sz(self):
        sz_squared = hams.sz @ hams.sz
        expected = 0.25 * hams.I
        assert np.allclose(sz_squared.todense(), expected.todense())

class Test_One_Spin_Operator:

    L = 3
    spin_op = hams.sy

    def test_one_spin_op_site0(self):
        idx = 0
        op = hams.one_spin_op(self.L, idx, self.spin_op)
        expected = kron(self.spin_op, kron(hams.I, hams.I))
        assert np.allclose(op.todense(), expected.todense())

    def test_one_spin_op_site1(self):
        idx = 1
        op = hams.one_spin_op(self.L, idx, self.spin_op)
        expected = kron(hams.I, kron(self.spin_op, hams.I))
        assert np.allclose(op.todense(), expected.todense())

    def test_one_spin_op_site2(self):
        idx = 2
        op = hams.one_spin_op(self.L, idx, self.spin_op)
        expected = kron(hams.I, kron(hams.I, self.spin_op))
        assert np.allclose(op.todense(), expected.todense())

    def test_one_spin_op_invalid_index_negative(self):
        idx = -1
        with pytest.raises(ValueError):
            hams.one_spin_op(self.L, idx, self.spin_op)
    
    def test_one_spin_op_invalid_index_too_large(self):
        idx = self.L
        with pytest.raises(ValueError):
            hams.one_spin_op(self.L, idx, self.spin_op)

class Test_Two_Spin_Operator:
    
    L = 5
    spin_op1 = hams.sx
    spin_op2 = hams.sy

    def test_two_spin_op_sites01(self):
        idx1 = 0
        idx2 = 1
        op = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        expected = kron(self.spin_op1, kron(self.spin_op2, kron(hams.I, kron(hams.I, hams.I))))
        assert np.allclose(op.todense(), expected.todense())

    def test_two_spin_op_sites24(self):
        idx1 = 2
        idx2 = 4
        op = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        expected = kron(hams.I, kron(hams.I, kron(self.spin_op1, kron(hams.I, self.spin_op2))))
        assert np.allclose(op.todense(), expected.todense())

    def test_two_spin_op_sites13(self):
        idx1 = 1
        idx2 = 5
        op = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        expected = kron(hams.I, kron(self.spin_op1, kron(hams.I, kron(self.spin_op2, hams.I))))
        assert np.allclose(op.todense(), expected.todense())

    def test_two_spin_op_sites_order(self):
        idx1 = 3
        idx2 = 1
        order_1 = hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
        order_2 = hams.two_spin_op(self.L, idx2, idx1, self.spin_op2, self.spin_op1)
        assert np.allclose(order_1.todense(), order_2.todense())

    def test_two_spin_op_same_site(self):
        idx1 = 2
        idx2 = 2
        with pytest.raises(ValueError):
            hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)

    # def test_two_spin_op_invalid_index_negative(self):
    #     idx1 = -1
    #     idx2 = 2
    #     with pytest.raises(ValueError):
    #         hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)

    # def test_two_spin_op_invalid_index_too_large(self):
    #     idx1 = 1
    #     idx2 = self.L
    #     with pytest.raises(ValueError):
    #         hams.two_spin_op(self.L, idx1, idx2, self.spin_op1, self.spin_op2)
            
# class Test_TFIM_Hamiltonian:

#     def test_zeros(self):
#         L = 4
#         J = 0
#         hx_arr = np.zeros(L)
#         ham = hams.tfim_hamiltonian(L, J, hx_arr, periodic=False)
#         expected = csr_matrix((2 ** L, 2 ** L), dtype=complex)
#         assert np.allclose(ham.todense(), expected.todense())

#     def test_manual_field_hamiltonian(self):
#         L = 5
#         J = 0.0
#         hx_arr = np.array([0.5, 0, 0, 1, 0])
#         ham = hams.tfim_hamiltonian(L, J, hx_arr, periodic=False)
#         expected = csr_matrix((2 ** L, 2 ** L), dtype=complex)
#         expected += hx_arr[0] * hams.one_spin_op(L, 0, hams.sx)
#         expected += hx_arr[3] * hams.one_spin_op(L, 3, hams.sx)
#         assert np.allclose(ham.todense(), expected.todense())

#     def test_coupling_terms_hamiltonian(self):
#         L = 3
#         J = 2.0
#         hx_arr = np.zeros(L)
#         ham = hams.tfim_hamiltonian(L, J, hx_arr, periodic=False)
#         expected = csr_matrix((2 ** L, 2 ** L), dtype=complex)
#         expected += J * hams.two_spin_op(L, 0, 1, hams.sz, hams.sz)
#         expected += J * hams.two_spin_op(L, 1, 2, hams.sz, hams.sz)
#         assert np.allclose(ham.todense(), expected.todense())


    # def test_periodic_boundary_conditions(self):
    #     L = 4
    #     J = 3.0
    #     hx_arr = np.ones(L)
    #     ham = hams.tfim_hamiltonian(L, J, hx_arr, periodic=True)
    #     expected = csr_matrix((2 ** L, 2 ** L), dtype=complex)
    #     expected += J * hams.two_spin_op(L, 0, 1, hams.sz, hams.sz)
    #     expected += J * hams.two_spin_op(L, 1, 2, hams.sz, hams.sz)
    #     expected += J * hams.two_spin_op(L, 2, 3, hams.sz, hams.sz)
    #     expected += J * hams.two_spin_op(L, 0, 3, hams.sz, hams.sz)
    #     for i in range(L):
    #         expected += hx_arr[i] * hams.one_spin_op(L, i, hams.sx)
    #     assert np.allclose(ham.todense(), expected.todense())

    # def test_invalid_hx_arr_length(self):
    #     L = 3
    #     J = 1.0
    #     hx_arr = np.array([0.5, 0.5])  # Incorrect length
    #     with pytest.raises(ValueError):
    #         hams.tfim_hamiltonian(L, J, hx_arr, periodic=False)
