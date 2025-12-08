from scipy.sparse import coo_array, identity, kron

sx = 0.5 * coo_array([[0, 1], [1, 0]])

sy = 0.5 * coo_array([[0, -1j], [1j, 0]])

sz = 0.5 * coo_array([[1, 0], [0, -1]])

I = identity(2)

def trace(matrix):
    """Calculates the trace of a matrix."""
    return matrix.diagonal().sum()

def one_spin_op(L, idx, Op):
    #single spin operator for chain of length L at site idx
    idx = int(idx)
    lsize = idx
    rsize = L - idx - 1
    lhs = identity(2 ** lsize)
    rhs = identity(2 ** rsize)
    return kron(lhs, kron(Op, rhs))

def two_spin_op(L, idx1, idx2, Op1, Op2):
    #two spin operator for chain of length L at sites idx1, idx2
    idx1 = int(idx1)
    idx2 = int(idx2)
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
