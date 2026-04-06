import taichi as ti  # type: ignore
import taichi.math as tm  # type: ignore
import numpy as np
from scipy import ndimage

from Utils import Renderer
from Utils import Solver

#================================#
#              Init              #
#================================#
ti.init(arch=ti.gpu)

# Simulation params
nx, ny = 100, 100
nt = 40000

view_scale = 5
gui_width = nx * view_scale
gui_height = ny * view_scale

# Output texture (temperature field)
image_field = ti.Vector.field(3, dtype=ti.f32, shape=(nx*view_scale, ny*view_scale))


# Create GUI
window = ti.ui.Window("LBM Thermal", (nx*view_scale, ny*view_scale))
canvas = window.get_canvas()
image_field = ti.Vector.field(3, dtype=ti.f32, shape=(nx*view_scale, ny*view_scale))
print(f"Résolution simulation: {nx}x{ny}")
print(f"Taille affichage: {gui_width}x{gui_height}")

# Fields
rho = ti.field(dtype=ti.f32, shape=(nx, ny))
ux = ti.field(dtype=ti.f32, shape=(nx, ny))
uy = ti.field(dtype=ti.f32, shape=(nx, ny))
T = ti.field(dtype=ti.f32, shape=(nx, ny))
f = ti.field(dtype=ti.f32, shape=(9, nx, ny))
g = ti.field(dtype=ti.f32, shape=(9, nx, ny))

# Constants
w = ti.Vector([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = ti.Vector.field(2, dtype=ti.i32, shape=9)
opp = ti.Vector([0,3,4,1,2,7,8,5,6])

c_list = [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]

for i in range(9):
    c[i] = c_list[i]

# Physics
tau_f, tau_g = 0.8, 0.99
omega_f, omega_g = 1.0/tau_f, 1.0/tau_g

beta = 1e-3
gravity = 1e-3
T0 = 2.0
u_left = 0.02
T_bottom = 2.2



# GUI
Solver.initialize(rho, T, ux, uy, f, g, T0, w)

# Fixe la plage d'affichage de la température pour éviter le scintillement
T_min_fixed = 1.8
T_max_fixed = 2.4

for step in range(nt):
    
    Solver.collide(rho, ux, uy, T, f, g, c, w, omega_f, omega_g, beta, gravity, T0)
    Solver.stream(f, g, c, rho, nx, ny)
    Solver.apply_bc(f, g, rho, w, T_bottom, u_left, nx, ny, opp)
    Solver.macroscopic(rho, ux, uy, T, f, g, c)

    Renderer.fill_image(image_field, T, T_min_fixed, T_max_fixed, view_scale)
    canvas.set_image(image_field)
    window.show()