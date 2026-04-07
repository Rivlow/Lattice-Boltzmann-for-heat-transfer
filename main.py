import taichi as ti
import taichi.math as tm
import numpy as np

from Utils import Renderer
from Utils import Solver
from Utils import BoundaryCondition

#================================#
#              Init              #
#================================#
ti.init(arch=ti.gpu)

# Numerical params
nx, ny = 500, 500
Lx, Ly = 1.0, 1.0  # taille physique du domaine (par ex. en mètre)
dx = Lx / nx
dy = Ly / ny

nt = 40000

x_min, x_max = 0.4, 0.6
y_min, y_max = 0.4, 0.6

i_min = int(x_min / dx)
i_max = int(x_max / dx)
j_min = int(y_min / dy)
j_max = int(y_max / dy)

# LBM params
tau_f, tau_g = 0.8, 0.99
omega_f, omega_g = 1.0/tau_f, 1.0/tau_g
beta = 1e-3

# Physical params
gravity = 1e-3
T0 = 2.0
u_left = 0.02

T_bottom = 2.2
T_min_fixed = 1.8
T_max_fixed = 2.4

# Output texture (temperature field)
view_scale = 1
gui_width = nx * view_scale
gui_height = ny * view_scale

# Create GUI (ti.GUI instead of ti.ui.Window)
gui = ti.GUI("LBM Thermal", res=(gui_width, gui_height))

print(f" Simulation meshing: {nx}x{ny}")
print(f"Pixels displayed: {gui_width}x{gui_height}")

# Fields
rho = ti.field(dtype=ti.f32, shape=(nx, ny))
ux = ti.field(dtype=ti.f32, shape=(nx, ny))
uy = ti.field(dtype=ti.f32, shape=(nx, ny))
U = ti.field(dtype=ti.f32, shape=(nx, ny))

T = ti.field(dtype=ti.f32, shape=(nx, ny))
f = ti.field(dtype=ti.f32, shape=(9, nx, ny))
g = ti.field(dtype=ti.f32, shape=(9, nx, ny))

image_field = ti.Vector.field(3, dtype=ti.f32, shape=(gui_width, gui_height))

# ------------ DEFINE OBSTACLES ------------------#
obstacles_physical = np.array([
    0.4, 0.6, 0.4, 0.6,   # obstacle central
    0.1, 0.2, 0.7, 0.8,   # autre obstacle
], dtype=np.float32)


# Constants
w = ti.Vector([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = ti.Vector.field(2, dtype=ti.i32, shape=9)
opp = ti.Vector([0,3,4,1,2,7,8,5,6])

c_list = [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
for i in range(9):
    c[i] = c_list[i]

#======================================#
#              SIMULATION              #
#======================================#

# Initialisation
Solver.initialize(rho, T, ux, uy, f, g, T0, w)

# Time loop
for step in range(nt):

    Solver.collide(rho, ux, uy, T, f, g, c, w, omega_f, omega_g, beta, gravity, T0)
    Solver.stream(f, g, c, rho, nx, ny)
    BoundaryCondition.apply_bc(f, g, rho, w, T_bottom, u_left, nx, ny, opp,
                            obstacles_physical, dx, dy)
    
    Solver.macroscopic(rho, ux, uy, T, f, g, c)
    Solver.compute_speed(U, ux, uy)

    Renderer.fill_image(image_field, U, 0, 0.05, view_scale)
    gui.set_image(image_field)

    gui.show()