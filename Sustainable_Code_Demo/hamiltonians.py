from scipy.sparse import coo_array, csr_matrix, identity, kron

sx = 0.5 * coo_array([[0, 1], [1, 0]]).tocsr()

sy = 0.5 * coo_array([[0, -1j], [1j, 0]]).tocsr()

sz = 0.5 * coo_array([[1, 0], [0, -1]]).tocsr()

I = identity(2)

def trace(matrix):
    """Calculates the trace of a matrix."""
    return matrix.diagonal().sum()

def one_spin_op(L, idx, Op):
    #single spin operator for chain of length L at site idx
    idx = int(idx)
    if idx < 0 or idx >= L:
        raise ValueError("idx must be between 0 and L-1.")
    lsize = idx
    rsize = L - idx - 1
    lhs = identity(2 ** lsize)
    rhs = identity(2 ** rsize)
    return kron(lhs, kron(Op, rhs))

def two_spin_op(L, idx1, idx2, Op1, Op2):
    #two spin operator for chain of length L at sites idx1, idx2
    idx1 = int(idx1)
    idx2 = int(idx2)
    if idx1 < 0 or idx1 >= L or idx2 < 0 or idx2 >= L:
        raise ValueError("idx1 and idx2 must be between 0 and L-1.")
    if idx1 == idx2:
        raise ValueError("idx1 and idx2 must be different.")
    lsize = min(idx1, idx2)
    msize = abs(idx2 - idx1) - 1
    rsize = L - max(idx1, idx2) - 1
    lhs = identity(2 ** lsize)
    mhs = identity(2 ** msize)
    rhs = identity(2 ** rsize)
    if idx1 < idx2:
        return kron(lhs, kron(Op1, kron(mhs, kron(Op2, rhs))))
    else:
        return kron(lhs, kron(Op2, kron(mhs, kron(Op1, rhs))))

def tfim_hamiltonian(L, J, hx_arr, periodic = False):
    # sparse Transverse-Field Ising Model Hamiltonian. Takes arrays for coupling Ji and fields hxi and hzi
    L = int(L)
    if hx_arr.size != L:
        raise ValueError("Length of hx_arr must be equal to L.")
    ham = csr_matrix((2 ** L, 2 ** L), dtype=complex)
    for i in range(L-1):
        ham += J * two_spin_op(L, i, i + 1, sz, sz)
    if periodic == True:
        # add coupling between first and last spin
        ham += J * two_spin_op(L, 0, L - 1, sz, sz)
    for i in range(L):
        ham += hx_arr[i] * one_spin_op(L, i, sx)
    return ham