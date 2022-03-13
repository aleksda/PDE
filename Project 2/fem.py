import numpy as np
import sympy as sym
from sympy.printing.latex import LatexPrinter, print_latex
import matplotlib.pyplot as plt

def mesh_uniform(N_e, d, Omega=[0,1]):

    if d == 1:

        start_nodes    =     N_e
        stop_nodes     = 2 * N_e * d + 1
        start_elements = 2 * N_e
        stop_elements  = 2 * N_e + N_e

    elif d == 2:

        start_nodes    = 0
        stop_nodes     = N_e * d + 1
        start_elements = 0
        stop_elements  = N_e
        
    elements = [[e*d + i for i in range(d+1)] \
                for e in range(start_elements, stop_elements)]

    h  = sym.Symbol('h')         # Length of the elements
    dx = h * sym.Rational(1, d)  # Spacing of nodes

    nodes = [Omega[0] + i * dx for i in range(start_nodes, stop_nodes)]

    if d == 1:
        nodes.remove(nodes[0])
        return nodes, elements

    elif d == 2:
        return nodes, elements

def P2P1(N_e):

    d1 = 1; P1_nodes, P1_elements = mesh_uniform(N_e, d = d1)
    d2 = 2; P2_nodes, P2_elements = mesh_uniform(N_e, d = d2)

    nodes   = P2_nodes    + P1_nodes
    dof_map = P2_elements + P1_elements

    cells    = []
    vertices = []

    for dof in dof_map:
        cells.append([dof[0], dof[-1]])
        vertices.append(nodes[dof[0]])

        if dof == dof_map[-1]:
            vertices.append(nodes[dof[-1]])

    return nodes, vertices, cells, dof_map
  

def Lagrange_polynomial(x, i, points):
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k])/(points[i] - points[k])
    return p


def Lagrange_polynomials(x, Omega_e, nodes):
    D = sym.Symbol("D") # Drichelet boundary conditions
    N = len(Omega_e)    
    points = [nodes[e] for e in Omega_e]
    psi = [Lagrange_polynomial(x, i, points) for i in range(N)]
    
    return psi, points

def loop_through(x, nodes, dof_map):

    N = np.concatenate(dof_map).ravel().shape[0] # The total number of mesh points

    psi_sym  = sym.zeros(N, 1)
    dpsi_sym = sym.zeros(N, 1)

    i = 0
    for dof in dof_map:

        psi, points= Lagrange_polynomials(x, Omega_e = dof, nodes = nodes)
        for j in range(len(dof)):

            psi_sym[i]  = psi[j]
            dpsi_sym[i] = sym.diff(psi[j], x)
            i += 1

    return psi_sym, dpsi_sym

def globalMatrix(x, dpsi, nodes, cells, dof_map):

    A = sym.zeros(len(nodes) - 1, len(nodes) - 1)

    i = 0
    for dof in dof_map:
        cell = cells[i]

        if i != len(dof_map) - 1:

            for j in range(len(dof)):
                for k in range(len(dof)):
                    integral = sym.integrate(dpsi[dof[j] + i] * dpsi[dof[k] + i], (x, nodes[cell[0]], nodes[cell[1]]))
                    A[dof[j], dof[k]] += integral

        elif i == len(dof_map) - 1:

            for j in range(len(dof) - 1):
                for k in range(len(dof) - 1):
                    integral = sym.integrate(dpsi[dof[j] + i] * dpsi[dof[k] + i], (x, nodes[cell[0]], nodes[cell[1]]))
                    A[dof[j], dof[k]] += integral

        i += 1

    return A

def globalVector(x, f, psi, dpsi, nodes, cells, dof_map):
    C = sym.Symbol("C") # Neumann   boundary conditions
    D = sym.Symbol("D") # Drichelet boundary conditions

    f = sym.sympify(f)

    b = sym.zeros(len(nodes) - 1, 1)

    i = 0
    for dof in dof_map:
        cell = cells[i]

        if i != 0 and i != len(dof_map) - 1:

            for j in range(len(dof)):
                b[dof[j]] += sym.integrate(f*psi[dof[j] + i], (x, nodes[cell[0]], nodes[cell[1]]))
                b[dof[j]]  = sym.simplify(b[dof[j]])

        elif i == 0:
            b[0] += sym.integrate(f*psi[0], (x, nodes[cell[0]], nodes[cell[1]])) - C
            b[1] += sym.integrate(f*psi[1], (x, nodes[cell[0]], nodes[cell[1]]))
            b[2] += sym.integrate(f*psi[2], (x, nodes[cell[0]], nodes[cell[1]]))

            for j in range(len(dof)):
                b[dof[j]] = sym.simplify(b[dof[j]])

        elif i == len(dof_map) - 1:

            b[dof[0]] += sym.integrate(f*psi[dof[0] + i], (x, nodes[cell[0]], nodes[cell[1]])) -\
                sym.integrate(D*dpsi[dof[1] + i] * dpsi[dof[0] + i], (x, nodes[cell[0]], nodes[cell[1]]))
            b[dof[0]] = sym.simplify(b[dof[0]])
  
        i += 1

    return b

def latex_nodes(dof_map, N_e):
    
    h = sym.Symbol('h') 
    N = 2 * N_e + 1 + N_e

    nodes = [i*h for i in range(N)]

    vertices = []
    for dof in dof_map:
        vertices.append(nodes[e[0]])
        if dof == dof_map[-1]:
            vertices.append(nodes[dof[-1]])
    
    return nodes, vertices

def solver(N_e, N, f, C_, D_):

    x = sym.Symbol("x")
    h = sympy.Symbol("h")
    C = sym.Symbol("C") # Neumann   boundary conditions
    D = sym.Symbol("D") # Drichelet boundary conditions
    
    nodes, vertices, cells, dof_map = P2P1(N_e)
    
    psi_sym, dpsi_sym = loop_through(x, nodes, dof_map)
    psi = psi_sym
    dpsi = dpsi_sym
    
    A = globalMatrix(x, dpsi, nodes, cells, dof_map)
    b = globalVector(x, f, psi, dpsi, nodes, cells, dof_map)
    
    print(nodes)
    step = 1 / (2 * N_e)
    nodes_sym = sym.Array(nodes)
    nodes_num = np.array(nodes_sym.subs(h, step)).astype(np.float64)


    A_num = np.array(A.subs({h:step, D:D_})).astype(np.float64)
    b_num = np.array(b.subs({h:step, C:C_, D:D_})).astype(np.float64)
    
    
    psi1  = psi
    dpsi1 = dpsi
    
    psi  = psi.subs(D, D_)
    dpsi = dpsi.subs(D, D_)
    
    c_num = np.linalg.pinv(A_num) @ b_num

    psi_num = [sym.lambdify([x, h], psi[i], modules= "numpy") for i in range(len(psi))]
    
    solution = np.zeros(N * len(dof_map))

    i = 0
    for dof in dof_map:
        sum_psi_e = 0
        cell = cells[i]
        x = np.linspace(nodes_num[cell[0]], nodes_num[cell[1]], N)

        if i != len(dof_map) - 1:

            for j in range(len(dof)):
                sum_psi_e += c_num[dof[j]] * psi_num[dof[j] + i](x, step)
                print(nodes[cell[0]], nodes[cell[1]], i, "|", dof[j], dof[j] + i)

        elif i == len(dof_map) - 1:

            sum_psi_e += c_num[dof[0]]*psi_num[dof[0] + i](x, step)
            sum_psi_e += psi_num[dof[1] + i](x, step)*D_

            for j in range(len(dof) - 1):
                print(nodes[cell[0]], nodes[cell[1]], j, "|", dof[j], dof[j] + i)
        
        solution[i*N :i*N + N] += sum_psi_e
        i += 1

    return solution, A, b, psi, dpsi, psi1, dpsi1, nodes, dof_map

#h = sym.Symbol("h")

x = sym.Symbol("x")

C = sym.Symbol("C") # Neumann   boundary condition
D = sym.Symbol("D") # Drichelet boundary condition

def u_exact1(x, C, D):
    return 0.5 * x**2 - (1/3) * x**3 + C * (x-1) + D - 1/6

def u_exact2(x, C, D):
    return (x**4) / 4 + x**2 + C * (x-1) + D - 5/4

#f  = "2*x - 1"
#rf = r"$2x - 1$"
f  = "-3*x**2 - 2"
rf = r"$-3x^2 -2$"

C_ = 1; D_ = 1

solution, A, b, psi, dpsi, psi1, dpsi1, nodes, dof_map = solver(N_e= 6, N= 1000, f = f, C_= C_, D_= D_)

x_num = np.linspace(0, 1, len(solution))
x     = np.linspace(0, 1)

analytical  = u_exact2(x, C_, D_)

plt.title(fr"$f(x) =$ {rf}, $C = {C_}$, $D = {D_}$")
plt.plot(x_num, solution, label= fr"numerical $N = {len(solution)}$")
plt.plot(x, analytical,   label= fr"analytical $N = {len(x)}$")
plt.xlabel(r"$x$"); plt.ylabel(r"$f(x)$")
plt.legend(loc="best")
plt.show()
"""