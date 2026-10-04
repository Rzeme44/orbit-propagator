import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from propagator import *

d = np.deg2rad
r0, v0 = coe2rv(6878, 0.001, d(51.6), d(60), d(30), d(0))
days = 2
t = np.linspace(0, days * 86400, 600)

sol = solve_ivp(two_body_j2, [0, t[-1]], np.concatenate((r0, v0)),
                t_eval=t, rtol=1e-10, atol=1e-10)

Om = np.unwrap([rv2coe(sol.y[:3, k], sol.y[3:, k])[3] for k in range(len(t))])
Om_deg = np.rad2deg(Om)
rate = (Om_deg[-1] - Om_deg[0]) / days
print("RAAN drift:", rate, "deg/day")

plt.plot(t / 86400, Om_deg)
plt.xlabel("time (days)"); plt.ylabel("RAAN (deg)")
plt.savefig("results/j2_raan.png", dpi=200)
plt.show()