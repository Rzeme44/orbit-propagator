import sys; sys.path.insert(0, ".")
import numpy as np
from scipy.integrate import solve_ivp
from propagator import *

d = np.deg2rad
a, e, inc = 6878, 0.001, 51.6
r0, v0 = coe2rv(a, e, d(inc), d(60), d(30), d(0))

days = 10
t = np.linspace(0, days * 86400, 3000)
sol = solve_ivp(two_body_j2, [0, t[-1]], np.concatenate((r0, v0)),
                t_eval=t, rtol=1e-10, atol=1e-10)

Om = np.rad2deg(np.unwrap([rv2coe(sol.y[:3, k], sol.y[3:, k])[3] for k in range(len(t))]))
numeric = np.polyfit(t / 86400, Om, 1)[0]   # slope in deg/day

n = np.sqrt(MU / a**3)
p = a * (1 - e**2)
analytic = np.rad2deg(-1.5 * n * J2 * (RE / p)**2 * np.cos(d(inc))) * 86400

print("numeric :", numeric, "deg/day")
print("analytic:", analytic, "deg/day")
print("error   :", abs(numeric - analytic) / abs(analytic) * 100, "%")