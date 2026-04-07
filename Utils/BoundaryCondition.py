import taichi as ti  
import taichi.math as tm

# -----------------------------
# Fonctions utilitaires
# -----------------------------

@ti.func
def coord_to_index(x: float, y: float, dx: float, dy: float, nx: int, ny: int):
    i = int(x / dx)
    j = int(y / dy)
    i = min(max(i, 0), nx - 1)
    j = min(max(j, 0), ny - 1)
    return i, j

@ti.func
def in_physical_zone(i: int, j: int, zone, dx: float, dy: float):
    x = (i + 0.5) * dx
    y = (j + 0.5) * dy
    xmin, xmax, ymin, ymax = zone
    return xmin <= x < xmax and ymin <= y < ymax

# -----------------------------
# Kernel apply_bc
# -----------------------------

@ti.kernel
def apply_bc(f: ti.template(), g: ti.template(), rho: ti.template(),
             w: ti.template(), T_bottom: float, u_left: float,
             nx: int, ny: int, opp: ti.template(),
             obstacles: ti.types.ndarray(), dx: float, dy: float):
    
    # Bords gauche / droite
    for j in range(ny):
        rho_left = (f[0,0,j] + f[2,0,j] + f[4,0,j] + 2*(f[3,0,j] + f[6,0,j] + f[7,0,j])) / (1 - u_left)
        f[1,0,j] = f[3,0,j] + 2/3 * rho_left * u_left
        f[5,0,j] = f[7,0,j] + 0.5*(f[4,0,j]-f[2,0,j]) + 1/6*rho_left*u_left
        f[8,0,j] = f[6,0,j] + 0.5*(f[2,0,j]-f[4,0,j]) + 1/6*rho_left*u_left

        # Extrapolation à droite
        f[3,nx-1,j] = f[3,nx-2,j]
        f[6,nx-1,j] = f[6,nx-2,j]
        f[7,nx-1,j] = f[7,nx-2,j]

    # Bords bas / haut (température imposée)
    for i in range(nx):
        # Bas
        f[2,i,0] = f[4,i,0]
        f[5,i,0] = f[7,i,0]
        f[6,i,0] = f[8,i,0]
        g[2,i,0] = -g[4,i,0] + 2*w[2]*T_bottom
        g[5,i,0] = -g[7,i,0] + 2*w[5]*T_bottom
        g[6,i,0] = -g[8,i,0] + 2*w[6]*T_bottom

        # Haut (symétrie / extrapolation)
        f[4,i,ny-1] = f[2,i,ny-1]
        f[7,i,ny-1] = f[5,i,ny-1]
        f[8,i,ny-1] = f[6,i,ny-1]
        g[4,i,ny-1] = g[2,i,ny-1]
        g[7,i,ny-1] = g[5,i,ny-1]
        g[8,i,ny-1] = g[6,i,ny-1]

    # Obstacles
    for i, j in rho:
        for z in range(obstacles.shape[0] // 4):
            xmin = obstacles[4*z + 0]
            xmax = obstacles[4*z + 1]
            ymin = obstacles[4*z + 2]
            ymax = obstacles[4*z + 3]
            zone = (xmin, xmax, ymin, ymax)
            if in_physical_zone(i, j, zone, dx, dy):
                for k in ti.static(range(9)):
                    f[k,i,j] = f[opp[k],i,j]
                    g[k,i,j] = g[opp[k],i,j]