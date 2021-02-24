import numpy as np
import pennylane as qml

from utils.linalg import so3_to_euler_angles, to_so3

class InvalidArgException(Exception): pass

def encode_3dvector(vec, wires):
    """
    Encode a 3d vector using angle encoding on 3 qubits
    """
    if len(wires) != 3:
        raise InvalidArgException("The `wires` argument must be passed exactly 3 qubits")
    qml.templates.AngleEmbedding(vec, wires, rotation='X')

def encode_unitary(mat, wires):
    """
    Encode 3x3 SO(3) matrix on two qubits, using SO(3) -> SU(2) embedding
    and Choi isomorphism
    """
    if len(wires) != 2:
        raise InvalidArgException("The `wires` argument must be passed exactly 2 qubits")

    phi, theta, psi = so3_to_euler_angles(mat)
    q0, q1 = wires

    # prepare Bell state
    qml.Hadamard(wires=q0)
    qml.CNOT(wires=[q0, q1])

    # perform unitary on first qubit
    qml.RZ(psi, wires=q0)
    qml.RX(theta, wires=q0)
    qml.RZ(phi, wires=q0)

def encode(cov, wires):
    """
    Given a covariance matrix, encode it in a circuit
    """
    eigs, vecs = np.linalg.eigh(cov)
    # make sure determinant is one
    vecs = to_so3(vecs)
    # consider log eigenvalues
    eigs = np.log(eigs)

    encode_3dvector(eigs, wires[:3])
    encode_unitary(vecs, wires[3:])