import numpy as np
import pennylane as qml

def encode_3dvector(vec, wires):
    """
    Encode a 3d vector using angle encoding on 3 qubits
    """
    qml.templates.AngleEmbedding(vec, wires, rotation='X')

def encode_unitary(mat, wires):
    """
    Encode 3x3 SO(3) matrix on two qubits, using SO(3) -> SU(2) embedding
    and Choi isomorphism
    """

    so3 = mat
    # using q = w + ix + jy + kz
    a = 0.5 * (so3[0, 0] - so3[1, 1]) # x^2 - y^2
    b = 0.5 * (so3[1, 1] - so3[2, 2]) # y^2 - z^2
    c = 0.5 * (so3[2, 2] - so3[1, 1]) # z^2 - x^2

    # => x^2 = y^2 + a
    x = np.sqrt()
    su2 = 

def encode(cov, wires):
    """
    Given a covariance matrix, encode it in a circuit
    """
    eigs, vecs = np.linalg.eig(cov)
    eigs = np.log(eigs)

    assert len(wires) >= 4
    encode_3dvector(eigs, wires[:3])
    encode_unitary(vecs, wires[2:4])