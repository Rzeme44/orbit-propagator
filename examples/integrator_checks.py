import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from propagator import *

r0 = np.array([6378.137 + 500, 0, 0])
v0 = np.array([0, np.sqrt(MU / np.linalg.norm(r0)), 0])
period = 2 * np.pi * np.sqrt(np.linalg.norm(r0)**3 / MU)
t = np.linspace(0, 3 * period, 2000)

for tol, name in [(1e-10, "tight"), (1e-4, "loose")]:
    sol = solve_ivp(two_body, [0, t[-1]], np.concatenate((r0, v0)),
                    rtol=tol, atol=tol, dense_output=True)
    Y = sol.sol(t)
    r, v = Y[:3], Y[3:]
    energy = 0.5 * np.sum(v**2, axis=0) - MU / np.linalg.norm(r, axis=0)

    plt.figure()
    plt.plot(t / 60, (energy - energy[0]) / abs(energy[0]))
    plt.xlabel("time (min)"); plt.ylabel("relative energy drift")
    plt.title(f"Energy drift, rtol={tol}")
    plt.savefig(f"results/energy_drift_{name}.png", dpi=200)

    plt.figure()
    plt.plot(sol.y[0], sol.y[1])
    plt.axis("equal"); plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.title(f"Orbit, rtol={tol}")
    plt.savefig(f"results/orbit_{name}.png", dpi=200)

plt.show()