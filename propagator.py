import numpy as np
from scipy.integrate import solve_ivp

MU = 398600.4418  # km^3/s^2

def two_body(t, state):
    r = state[:3]
    v = state[3:]
    a = -MU * r / np.linalg.norm(r)**3
    return np.concatenate((v, a))

def rv2coe(r, v, mu=MU):
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)
    e_vec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    a = -mu / (2 * (vn**2 / 2 - mu / rn))
    i = np.arccos(h[2] / np.linalg.norm(h))
    Om = np.arccos(n[0] / np.linalg.norm(n))
    if n[1] < 0: Om = 2 * np.pi - Om
    w = np.arccos(np.dot(n, e_vec) / (np.linalg.norm(n) * e))
    if e_vec[2] < 0: w = 2 * np.pi - w
    nu = np.arccos(np.dot(e_vec, r) / (e * rn))
    if np.dot(r, v) < 0: nu = 2 * np.pi - nu
    return a, e, i, Om, w, nu

def coe2rv(a, e, i, Om, w, nu, mu=MU):
    p = a * (1 - e**2)
    r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])
    def Rz(t): return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    def Rx(t): return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    R = Rz(Om) @ Rx(i) @ Rz(w)
    return R @ r_pf, R @ v_pf

J2 = 1.08263e-3
RE = 6378.137  # km

def two_body_j2(t, state):
    r = state[:3]
    v = state[3:]
    rn = np.linalg.norm(r)
    a = -MU * r / rn**3
    k = 1.5 * J2 * MU * RE**2 / rn**4
    z2 = (r[2] / rn)**2
    a_j2 = k * np.array([
        r[0] / rn * (5 * z2 - 1),
        r[1] / rn * (5 * z2 - 1),
        r[2] / rn * (5 * z2 - 3),
    ])
    return np.concatenate((v, a + a_j2))


def hohmann(r1, r2, mu=MU):
    a_t = (r1 + r2) / 2
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    vp = np.sqrt(mu * (2 / r1 - 1 / a_t))   # speed at start of ellipse
    va = np.sqrt(mu * (2 / r2 - 1 / a_t))   # speed at far end of ellipse
    tof = np.pi * np.sqrt(a_t**3 / mu)
    return vp - v1, v2 - va, tof

from scipy.optimize import brentq

def _C(z):
    if z > 0: return (1 - np.cos(np.sqrt(z))) / z
    if z < 0: return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 0.5

def _S(z):
    if z > 0: s = np.sqrt(z); return (s - np.sin(s)) / s**3
    if z < 0: s = np.sqrt(-z); return (np.sinh(s) - s) / s**3
    return 1 / 6

def lambert(r1, r2, dt, mu=MU):
    """Prograde, single-revolution Lambert solver (universal variables)."""
    r1n, r2n = np.linalg.norm(r1), np.linalg.norm(r2)
    dnu = np.arccos(np.dot(r1, r2) / (r1n * r2n))
    if np.cross(r1, r2)[2] < 0:
        dnu = 2 * np.pi - dnu
    A = np.sin(dnu) * np.sqrt(r1n * r2n / (1 - np.cos(dnu)))

    y = lambda z: r1n + r2n + A * (z * _S(z) - 1) / np.sqrt(_C(z))
    F = lambda z: (y(z) / _C(z))**1.5 * _S(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * dt

    z_low = -50.0
    while y(z_low) < 0:
        z_low += 0.1
    z = brentq(F, z_low + 1e-6, 4 * np.pi**2 - 1e-6)

    yz = y(z)
    f = 1 - yz / r1n
    g = A * np.sqrt(yz / mu)
    gdot = 1 - yz / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2

