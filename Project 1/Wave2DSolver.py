import numpy as np
import matplotlib.pyplot as plt

class Wave2DSolver():
    def __init__(self, params):
        
        self.b  = params["b"] # Damping coefficient
        self.Nx = params["Nx"]; self.Ny = params["Ny"] # Points in x and y direction
        self.Lx = params["Lx"]; self.Ly = params["Ly"] # Length in x and y direction
        self.T  = params["T"] # Simulation time
        
        self.x  = np.linspace(0, self.Lx, self.Nx); self.y = np.linspace(0, self.Ly, self.Ny)
        self.dx = self.x[1] - self.x[0]; self.dy = self.y[1] - self.y[0]
        
        # Variables to hold the values to the solutions
        x_start = 1; x_end = self.Nx + 1; y_start = 1; y_end = self.Ny + 1
        self.x_start = x_start; self.x_end = x_end; self.y_start = y_start; self.y_end = y_end
        
        #self.beta = 0.9
        self.beta = params["beta"] 
        
        #Condition functions
        self.U  = params["U"]
        self.I  = params["I"]
        self.V  = params["V"]
        self.f  = params["f"]
        self.q_ = params["q"]
        self.q  = self.__q_fill()
        
        self.Nt = int(round(self.T/self.dt))

        self.u   = np.zeros([x_end+1,y_end+1]) # making room for ghost values     #==>  # solution array l       =  1, 2, 3..
        self.u_1 = np.zeros([x_end+1,y_end+1]) # end + 1 = N + 2                  #==>  # solution at t - dt l   =  0, 1, 2..
        self.u_2 = np.zeros([x_end+1,y_end+1])                                    #==>  # solution at t - 2*dt l = -1, 0, 1..
        
        # Constants used in advance_scheme
        self.denom = 2 + self.b*self.dt
        self.inverse = 1/self.denom
        self.E = 1 - 4*self.inverse
        self.F = self.inverse
        
        self.title = "plot"
        self.x_l   = "x"
        self.y_l   = "y"
        
    def __initialize(self):
        
        self.__initial_u_old()
        self.__update_ghost_cells(self.u_2)
        
        self.__q_fill()
        #self.__stabilize()
        
        self.dt2 = self.dt**2            # Reduce FLOPs
        self.dampdt = self.b*0.5*self.dt # damping x dt/2
        
        #Constants used in first_step
        self.A  = 3 - self.dampdt
        self.Cx = self.dt / self.dx; self.B = 0.25*self.Cx**2
        self.Cy = self.dt / self.dy; self.C = 0.25*self.Cy**2
        self.D  = 0.5*self.dt2
        
        self.__first_timestep()
        self.__update_ghost_cells(self.u_1)
        
    def __initial_u_old(self):
        for i in range(1, self.x_end):
            for j in range(1, self.y_end):
                self.u_2[i,j] = self.I(self.x[i-1], self.y[j-1]) #self.I_func(self.x[i], self.y[j])
        
    def __update_ghost_cells(self, u):
        
        for i in range(1, self.x_end):
            u[i,0]  = u[i,2]
            u[i,-1] = u[i,-3]
            # u[i, Ny+1] = u[i, Ny-1]
            # u[i, y_end] = u[i, y_end-2]
        
        for j in range(1, self.y_end):
            u[0,j]  = u[2,j]
            u[-1,j] = u[-3,j]
            # u[Nx+1, j] = u[Nx-1, j]
            # u[x_end, j] = u[x_end-2, j]
        
    def __q_fill(self):
        Nx = self.Nx; Ny = self.Ny
        Q = np.zeros([self.x_end + 1, self.y_end + 1])
        # x_end = self.x_end; y_end = self.y_end
        
        # Values 1 - N
        for i in range(1, self.x_end):
            for j in range(1, self.y_end):
                Q[i,j]= self.q_(self.x[i-1], self.y[j-1]) # self.q_(self.x[i], self.y[j])
        
        self.c = np.sqrt(np.max(Q))
        # self.c = np.sqrt(np.max(abs(Q)))
        
        # Ghost values
        for i in range(1, self.x_end):
            Q[i, 0] = 2*Q[i, 1] - Q[i, 2]
            Q[i, Ny+1] = 2*Q[i, Ny] - Q[i, Ny-1]
        
        for j in range(1, self.y_end):
            Q[0, j] = 2*Q[1, j] - Q[2, j]
            Q[Nx+1, j] = 2*Q[Nx, j] - Q[Nx-1, j]
        
        self.__stabilize()
        # self.__stabilize2()
        return Q
        
    def __stabilize(self):
        self.dt = (1/np.sqrt( (1/self.dx**2) + (1/self.dy**2) ))*(1/self.c)*self.beta
        
    def __stabilize2(self):
        num     = self.dx * self.dy
        l2      = np.sqrt(self.dx**2 + self.y**2)
        self.dt = self.beta * num/self.c / l2
        
    def __first_timestep(self): # l = 0 --> t = 0
        q = self.q; u_2 = self.u_2
        A = self.A; B = self.B; C= self.C; D = self.D
        
        for i in range(1, self.x_end):
            for j in range(1, self.y_end):
                self.u_1[i,j] = A*self.dt*self.V(self.x[i-1], self.y[j-1]) + u_2[i,j] +\
                B*( (q[i,j] + q[i+1,j])*(u_2[i+1,j] - u_2[i,j]) - (q[i-1,j] + q[i,j])*(u_2[i,j] - u_2[i-1,j]) ) +\
                C*( (q[i,j] + q[i,j+1])*(u_2[i,j+1] - u_2[i,j]) - (q[i,j-1] + q[i,j])*(u_2[i,j] - u_2[i,j-1]) ) +\
                D*self.f(self.x[i-1], self.y[j-1], 0)
        
    def __advance_scheme(self):
        q = self.q; u_1 = self.u_1; u_2 = self.u_2
        E = self.E;   F = self.F;    Cx = self.Cx; Cy = self.Cy
        
        for i in range(1, self.x_end):
            for j in range(1, self.y_end):
                self.u[i,j] = E*u_2[i,j] + F*4*u_1[i,j] +\
                F*Cx**2*( (q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - (q[i-1,j] + q[i,j])*(u_1[i,j] - u_1[i-1,j]) ) +\
                F*Cy**2*( (q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - (q[i,j-1] + q[i,j])*(u_1[i,j] - u_1[i,j-1]) ) +\
                2*self.dt2*self.f(self.x[i-1], self.y[j-1], self.t)

    def __swap(self):
        self.u_2, self.u_1, self.u = self.u_1, self.u, self.u_2
        #self.u_2 = self.u_1
        #self.u_1 = self.u
        #self.u   = self.u_2

    @property
    def solve(self):
        self.__initialize()
        self.t = self.dt
        
        while self.t <= self.T:
            self.__advance_scheme()
            self.__update_ghost_cells(self.u)
            self.__swap()
            self.t += self.dt

    def error(self, u_exact):
        analytical_solution = np.zeros([self.Nx, self.Ny])

        u_exact_func = lambda x, y, t: u_exact(x, y, t)
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                analytical_solution[i,j] = u_exact_func(self.x[i], self.y[j], self.t)
        
        eps = analytical_solution - self.u[1:self.x_end, 1:self.y_end]
        return np.max(np.abs(eps)), np.min(np.abs(eps)), analytical_solution

    def error2(self, u_exact):
        analytical_solution = np.zeros((self.Nx, self.Ny))
        u_exact_func = lambda x, y, t: u_exact(x, y, t)
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                analytical_solution[i,j] = u_exact_func(self.x[i], self.y[j], self.t)
        
        eps = analytical_solution - self.u_1[1:self.x_end, 1:self.y_end]
        return np.max(np.abs(eps)), analytical_solution
        
    def print_matrix(self, A):
        print('\n'.join([''.join([' {:.4f}'.format(item) for item in row]) for row in A]), '\n')

    @property
    def plot_name(self):
        return self.title

    @plot_name.setter
    def plot_name(self, title): # = "plot"
        self.title = title

    @property
    def x_label_name(self):
        return self.x_l

    @x_label_name.setter
    def x_label_name(self, x_l): # = "x"
        self.x_l = x_l

    @property
    def y_label_name(self):
        return self.y_l

    @y_label_name.setter
    def y_label_name(self, y_l): # = "y"
        self.y_l = y_l

    @property
    def contour_plot(self):
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.xx = self.xx.T; self.yy = self.yy.T
        plt.contourf(self.xx, self.yy, self.u[1:self.x_end, 1:self.y_end])
        plt.title(self.title)
        plt.xlabel(self.x_l)
        plt.ylabel(self.y_l)
        plt.colorbar()
        plt.figure()
