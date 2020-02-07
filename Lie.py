#!/usr/bin/env python3

from math import sin, cos
import numpy as np
from scipy.linalg import expm, logm


def skew(omega):
    return np.array([[0, -omega[2].item(), omega[1].item()],
                     [omega[2].item(), 0, -omega[0].item()],
                     [-omega[1].item(), omega[0].item(), 0]])


def unskew(Omega):
    return np.array([[Omega[2, 1].item()], [Omega[0, 2].item()], [Omega[1, 0].item()]])


e1 = np.array([1, 0, 0]).transpose()
e2 = np.array([0, 1, 0]).transpose()
e3 = np.array([0, 0, 1]).transpose()
G1 = np.concatenate((np.concatenate((skew(e1),
                                     np.zeros((3, 1))), axis=1),
                     np.zeros((1, 4))), axis=0)
G2 = np.concatenate((np.concatenate((skew(e2),
                                     np.zeros((3, 1))), axis=1),
                     np.zeros((1, 4))), axis=0)
G3 = np.concatenate((np.concatenate((skew(e3),
                                     np.zeros((3, 1))), axis=1),
                     np.zeros((1, 4))), axis=0)
G4 = np.zeros((4, 4))
G4[0, 3] = 1
G5 = np.zeros((4, 4))
G5[1, 3] = 1
G6 = np.zeros((4, 4))
G6[2, 3] = 1
e = [e1, e2, e3]
G = [G1, G2, G3, G4, G5, G6]


def hat(x):
    X = np.zeros((4, 4))
    X[:3, :3] = skew(x[0:3])
    X[:3, [3]] = x[3:6]
    return X


def vee(X):
    x = np.zeros((6, 1))
    x[:3] = unskew(X[:3, :3])
    x[3:6] = X[:3, [3]]
    return x


def Adjoint_SO3(Omega):
    return Omega


def Adjoint_SE3(X):
    ret = np.zeros((6, 6))
    ret[:3, :3] = X[:3, :3]
    ret[3:6, :3] = skew(X[0:3, 3]).dot(X[:3, :3])
    ret[3:6, 3:6] = X[:3, :3]
    return ret


def LeftJacobian_SO3(w):
    theta = np.linalg.norm(w)
    A = skew(w)
    if theta == 0:
        return np.eye(3)
    # Note: when theta is small, we should use taylor series
    return np.eye(3) + ((1-cos(theta))/theta**2)*A + ((theta-sin(theta))/theta**3)*np.matmul(A, A)


def LeftJacobian_SE3(xi):
    Phi = xi[0:3]
    Rho = xi[3:6]
    phi = np.linalg.norm(Phi)
    Phi_skew = skew(Phi)
    Rho_skew = skew(Rho)
    J = LeftJacobian_SO3(Phi)
    if phi == 0:
        Q = 0.5*Rho_skew
    else:
        Q = 0.5 * Rho_skew
        + (phi-sin(phi))/phi**3 * (Phi_skew.dot(Rho_skew) +
                                   Rho_skew.dot(Phi_skew) +
                                   Phi_skew.dot(Rho_skew).dot(Phi_skew))
        - (1-0.5*phi**2-cos(phi))/phi**4 * (Phi_skew.dot(Phi_skew).dot(Rho_skew)
                                            + Rho_skew.dot(Phi_skew).dot(Phi_skew)
                                            - 3*Phi_skew.dot(Rho_skew).dot(Phi_skew))
        - 0.5*((1-0.5*phi**2-cos(phi))/phi**4 -
               3*(phi-sin(phi)-(phi**3)/6)/phi**5) * (Phi_skew.dot(Rho_skew).dot(Phi_skew).dot(Phi_skew)
                                                      + Phi_skew.dot(Phi_skew).dot(Rho_skew).dot(Phi_skew))
    ret_J = np.zeros((6,6))
    ret_J[:3,:3] = J
    ret_J[3:6,0:3] = Q
    ret_J[3:6,3:6] = J
    return ret_J


def RightJacobian_SO3(xi):
    return Adjoint_SO3(expm(skew(-xi))).dot(LeftJacobian_SO3(xi))


def RightJacobian_SE3(xi):
    return Adjoint_SE3(expm(hat(-xi))).dot(LeftJacobian_SE3(xi))


def RightJacobianInverse_SE3(xi):
    Jr = RightJacobian_SE3(xi)
    return np.linalg.inv(Jr)


def from_rpy_xyz(roll, pitch, yaw, x, y, z):
    rot = from_rpy(roll, pitch, yaw)
    rot_t = np.concatenate((rot, np.array([[x], [y], [z]])), axis=1)
    return np.concatenate((rot_t, np.array([[0, 0, 0, 1]])), axis=0)


def from_rpy(roll, pitch, yaw):
    return rotz(yaw).dot(roty(pitch).dot(rotx(roll)))


def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
