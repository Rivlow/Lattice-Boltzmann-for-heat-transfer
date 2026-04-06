import taichi as ti  
import taichi.math as tm

# Init
@ti.kernel
def initialize(rho: ti.template(), T: ti.template(),
               ux: ti.template(), uy: ti.template(),
               f: ti.template(), g: ti.template(),
               T0: float, w: ti.template()):
    for i, j in rho:
        rho[i, j] = 1.0
        ux[i, j] = 0.0
        uy[i, j] = 0.0
        T[i, j] = T0

    for i, j in rho:
        for k in ti.static(range(9)):
            f[k, i, j] = w[k]
            g[k, i, j] = w[k]

# Collision
@ti.kernel
def collide(rho: ti.template(), ux: ti.template(), uy: ti.template(),
            T: ti.template(), f: ti.template(), g: ti.template(),
            c: ti.template(), w: ti.template(),
            omega_f: float, omega_g: float,
            beta: float, gravity: float, T0: float):
    
    for i, j in rho:
        Fx = 0.0
        Fy = rho[i, j] * beta * (T[i, j] - T0) * gravity

        for k in ti.static(range(9)):
            cu = c[k][0]*ux[i,j] + c[k][1]*uy[i,j]

            feq = w[k] * rho[i,j] * (1 + 3*cu + 4.5*cu*cu - 1.5*(ux[i,j]**2 + uy[i,j]**2))

            F = (1 - 0.5*omega_f) * w[k] * (
                ((c[k][0] - ux[i,j]) * 3 + 9*cu*c[k][0]) * Fx +
                ((c[k][1] - uy[i,j]) * 3 + 9*cu*c[k][1]) * Fy
            )

            f[k, i, j] += -omega_f * (f[k, i, j] - feq) + F

            geq = w[k] * T[i,j] * (1 + 3*cu)
            g[k, i, j] += -omega_g * (g[k, i, j] - geq)

# Streaming
@ti.kernel
def stream(f: ti.template(), g: ti.template(), c: ti.template(), rho: ti.template(), nx: int, ny: int):
    for k in ti.static(range(9)):
        for i, j in rho:
            ip = (i - c[k][0]) % nx
            jp = (j - c[k][1]) % ny

            f[k, i, j] = f[k, ip, jp]
            g[k, i, j] = g[k, ip, jp]

# Boundary + Obstacle
@ti.kernel
def apply_bc(f: ti.template(), g: ti.template(), rho: ti.template(),
             w: ti.template(), T_bottom: float, u_left: float,
             nx: int, ny: int, opp: ti.template()):
    for j in range(ny):
        rho_left = (f[0,0,j] + f[2,0,j] + f[4,0,j] + 2*(f[3,0,j] + f[6,0,j] + f[7,0,j])) / (1 - u_left)
        f[1,0,j] = f[3,0,j] + 2/3 * rho_left * u_left
        f[5,0,j] = f[7,0,j] + 0.5*(f[4,0,j]-f[2,0,j]) + 1/6*rho_left*u_left
        f[8,0,j] = f[6,0,j] + 0.5*(f[2,0,j]-f[4,0,j]) + 1/6*rho_left*u_left

        f[3,nx-1,j] = f[3,nx-2,j]
        f[6,nx-1,j] = f[6,nx-2,j]
        f[7,nx-1,j] = f[7,nx-2,j]

    for i in range(nx):
        f[2,i,0] = f[4,i,0]
        f[5,i,0] = f[7,i,0]
        f[6,i,0] = f[8,i,0]
        g[2,i,0] = -g[4,i,0] + 2*w[2]*T_bottom
        g[5,i,0] = -g[7,i,0] + 2*w[5]*T_bottom
        g[6,i,0] = -g[8,i,0] + 2*w[6]*T_bottom

        f[4,i,ny-1] = f[2,i,ny-1]
        f[7,i,ny-1] = f[5,i,ny-1]
        f[8,i,ny-1] = f[6,i,ny-1]
        g[4,i,ny-1] = g[2,i,ny-1]
        g[7,i,ny-1] = g[5,i,ny-1]
        g[8,i,ny-1] = g[6,i,ny-1]

    for i, j in rho:
        if 40 <= i < 60 and 40 <= j < 60:
            for k in ti.static(range(9)):
                f[k,i,j] = f[opp[k],i,j]
                g[k,i,j] = g[opp[k],i,j]

# Macroscopic
@ti.kernel
def macroscopic(rho: ti.template(), ux: ti.template(), uy: ti.template(),
                T: ti.template(), f: ti.template(), g: ti.template(),
                c: ti.template()):
    for i, j in rho:
        r = 0.0
        ux_loc = 0.0
        uy_loc = 0.0
        temp = 0.0
        for k in ti.static(range(9)):
            val = f[k,i,j]
            r += val
            ux_loc += c[k][0]*val
            uy_loc += c[k][1]*val
            temp += g[k,i,j]
        rho[i,j] = r
        ux[i,j] = ux_loc / r
        uy[i,j] = uy_loc / r
        T[i,j] = temp