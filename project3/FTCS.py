
import numpy as np

class FTCS:

    def __init__(self, dx, L):
        self.L = L #length of the space interval
        self.dx = dx #stepsize for the spacial part
        self.dt = (self.dx**2) / 2 #stepsize for the time
        self.n = int(1/self.dx) + 1 #number of point for the spacial coordinate
        self.x = np.linspace(0, self.L, self.n) #vector with the space coordinates
        self.func = self.initial_f() #the initial space coordinate
        self.t = 0  

    #initial condition
    def initial_f(self):
        initial_vec = np.sin(np.pi * self.x)

        #Dirichlet boundary condition
        initial_vec[0], initial_vec[-1] = 0, 0
        return initial_vec

    #analytical solution for given t value
    def analytic(self, t):
        return np.sin(np.pi * self.x) * np.exp(-t * np.pi**2)

    #evolving FTCS
    def evolve_scheme_diffusion(self, t_final):
        self.t = 0
        self.func = self.initial_f().copy()
        steps = int(t_final / self.dt)
        u = self.func.copy()
        alpha = self.dt / self.dx**2
        MSE_history = np.zeros(steps)

        #looping over the time steps
        for j in range(steps):
            u_new = u.copy()

            #looping over steps in space
            for i in range(1, self.n - 1):
                u_new[i] = u[i] + alpha * (u[i+1] - 2 * u[i] + u[i-1])
            
            #Dirichlet boundary condition
            u_new[0], u_new[-1] = 0, 0

            u = u_new.copy()
            self.t += self.dt 

            #saving MSE at each point in time
            MSE_history[j] = self.MSE(self.t, u)
 
        self.func = u
        return u, MSE_history
    
    #Calculating mean square error
    def MSE(self, t, numerical_solution):
        analytical_solution = self.analytic(t)
        mse = np.mean((numerical_solution - analytical_solution) ** 2)
        return mse
