import numpy as np

epsilon = 1e-7

def _get_so3(phi, theta, psi):
    a11 = np.cos(phi)*np.cos(psi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    a12 = -np.cos(phi)*np.sin(psi) - np.cos(theta)*np.sin(phi)*np.cos(psi)
    a13 = np.sin(phi)*np.sin(theta)
    a21 = np.sin(phi)*np.cos(psi) + np.cos(theta)*np.cos(phi)*np.sin(psi)
    a22 = -np.sin(phi)*np.sin(psi) + np.cos(theta)*np.cos(phi)*np.cos(psi)
    a23 = - np.cos(phi)*np.sin(theta)
    a31 = np.sin(psi) * np.sin(theta)
    a32 = np.cos(psi) * np.sin(theta)
    a33 = np.cos(theta)
    return np.array([
        [a11, a12, a13],
        [a21, a22, a23],
        [a31, a32, a33]
    ])

def _get_su2(phi, theta, psi):
    a11 = np.cos(theta/2) * np.exp(1j * (phi+psi)/2)
    a12 = 1j*np.sin(theta/2) * np.exp(1j * (phi-psi)/2)
    a21 = 1j*np.sin(theta/2) * np.exp(-1j * (phi-psi)/2)
    a22 = np.cos(theta/2) * np.exp(-1j * (phi+psi)/2)
    return np.array([
        [a11, a12],
        [a21, a22]
    ])

def so3_to_euler_angles(so3):
    """ Given a 3x3 SO(3) matrix, compute its euler angles """
    assert np.abs(np.linalg.det(so3) - 1) < epsilon

    theta = np.arccos(so3[2, 2])
    phi   = np.arccos(- so3[1, 2] / np.sin(theta))
    psi   = np.arccos(so3[2, 1] / np.sin(theta))

    # fix sign
    if so3[2, 0] / np.sin(theta) < 0:
        psi *= -1
    if so3[0, 2] / np.sin(theta) < 0:
        phi *= -1
    return phi, theta, psi

def so3_to_su2(so3):
    """ Given a 3x3 SO(3) matrix, compute its image in SU(2)/{±1} ≈ SO(3) """
    phi, theta, psi = so3_to_euler_angles(so3)
    assert np.linalg.norm(so3 - _get_so3(phi, theta, psi)) < epsilon

    return _get_su2(phi, theta, psi)

def to_so3(mat):
    """ Normalises determinant of 3x3 matrix in O(3) to be in SO(3) """
    return mat * [1., 1., 1/np.linalg.det(mat)]