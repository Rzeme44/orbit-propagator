import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from propagator import *

r1, r2 = 6878, 42164   # LEO (500 km) to GEO
dv1, dv2, tof = hohmann(r1, r2)
print("dv1:", dv1, "km/s")
print("dv2:", dv2, "km/s")
print("total:", dv1 + dv2, "km/s")
print("time of flight:", tof / 3600, "hours")

v1 = np.sqrt(MU / r1)
state0 = np.array([r1, 0, 0, 0, v1 + dv1, 0])
sol = solve_ivp(two_body, [0, tof], state0, rtol=1e-10, atol=1e-10)
print("final radius:", np.linalg.norm(sol.y[:3, -1]), "target:", r2)

th = np.linspace(0, 2 * np.pi, 400)
plt.plot(r1 * np.cos(th), r1 * np.sin(th), label="LEO")
plt.plot(r2 * np.cos(th), r2 * np.sin(th), label="GEO")
plt.plot(sol.y[0], sol.y[1], label="transfer")
plt.axis("equal"); plt.legend()
plt.xlabel("x (km)"); plt.ylabel("y (km)")
plt.savefig("results/hohmann.png", dpi=200)
plt.show()