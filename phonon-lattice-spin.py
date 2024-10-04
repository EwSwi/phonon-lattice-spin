import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# just some holders untill i will figure out the full picture
mu = 1.0
omega_0 = 1.0
rho = 0.5
beta = 0.2
epsilon = lambda t: np.sin(t)  
Sx = lambda t: np.cos(t)  
Sy = lambda t: np.sin(t)  
Sz = lambda t: 0.5 * np.sin(t)  
# holders end


# 1st order
def phonon_system(t, y):
    u, v = y
    spin_magnitude = np.sqrt(Sx(t)**2 + Sy(t)**2 + Sz(t)**2)
    du_dt = v
    dv_dt = mu * spin_magnitude - omega_0**2 * u - 2 * omega_0 * rho * epsilon(t) * u - rho**2 * epsilon(t)**2 * u - 2 * beta * v
    return [du_dt, dv_dt]

def omega_correction(t):
    return omega_0 + rho * epsilon(t)

# [u(0), v(0)]
initial_conditions = [0, 0]

#time evolution
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

#solver
solution = solve_ivp(phonon_system, t_span, initial_conditions, t_eval=t_eval)
omega_correction_values = omega_correction(t_eval)

print("omega correction:", omega_correction_values)
# plot
plt.plot(solution.t, solution.y[0], label="u(t)")
plt.xlabel("Time")
plt.ylabel("u(t)")
plt.legend()
plt.title("Numerical Solution of Phonon Damping with LLG Influence")
plt.show()
