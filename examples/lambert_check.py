import sys; sys.path.insert(0, ".")
import numpy as np
from scipy.integrate import solve_ivp
from propagator import *

r1 = np.array([5000.0, 10000.0, 2100.0])
r2 = np.array([-14600.0, 2500.0, 7000.0])
dt = 3600.0

v1, v2 = lambert(r1, r2, dt)
print("v1:", v1)
print("v2:", v2)

# Independent check: propagate with the integrator and see if we land on r2
sol = solve_ivp(two_body, [0, dt], np.concatenate((r1, v1)), rtol=1e-10, atol=1e-10)
print("miss distance:", np.linalg.norm(sol.y[:3, -1] - r2), "km")