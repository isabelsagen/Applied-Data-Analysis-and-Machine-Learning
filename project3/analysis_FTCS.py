
from FTCS import FTCS
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.size'] = 12

dx = 0.01
L = 1
t_values = [0.02, 0.2, 0.4, 0.7]


ftcs = FTCS(dx, L)



#Plotting the solution of the diffusion equation
fig, axs = plt.subplots(2, 2)
for idx, t in enumerate(t_values):
    numerical_solution, MSE_list = ftcs.evolve_scheme_diffusion(t)
    analytical_solution = ftcs.analytic(t)
    mse = MSE_list[-1]

    
    ax = axs[idx // 2, idx % 2]
    ax.plot(ftcs.x, numerical_solution, label="Numerical")
    ax.plot(ftcs.x, analytical_solution, label="Analytical", linestyle="dashed")
    ax.text(0.2, 0.4, f"MSE = {mse:.2e}")
    ax.set_ylim(-0.1, 1)
    ax.legend()
    ax.grid()
    ax.set_title(f"t = {t}")

fig.text(0.5, 0.04, "x", ha="center")
fig.text(0.04, 0.5, "u(x)", va="center", rotation="vertical")
plt.tight_layout()
plt.savefig("FTCS_001.pdf", format="pdf")
plt.show()

#Plotting MSE over time for dx = 0.1, 0.01
L = 1
T = 1

ftcs = FTCS(0.1, L)
numerical_solution, MSE_01 = ftcs.evolve_scheme_diffusion(T)

ftcs = FTCS(0.01, L)
numerical_solution, MSE_001 = ftcs.evolve_scheme_diffusion(T)

plt.plot(np.linspace(0, T, len(MSE_01)), MSE_01, label = r'$\Delta x =0.1$')
plt.plot(np.linspace(0, T, len(MSE_001)), MSE_001, label = r'$\Delta x =0.01$')
plt.yscale('log')
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("MSE")
plt.grid()
plt.savefig("FTCS_MSE.pdf", format="pdf")
plt.show()

