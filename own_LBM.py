import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os
import random


###################################
#  Lattice-Boltzmann method (2D)  #
###################################
        

#=====================#
# Physical parameters #
#=====================#

# SI units
kg, m, s = 1, 1, 1

c1 = 1  
rho_0 = 1
gravity = 1e-3 #9.81 * (m/s**2)

c_s2 = 1/3 # TO SO !!! use dx/dy/dt (here, all set to 1 )
c_s4 = 1/9

tau_f, tau_g = 0.8, 0.99 # Collision timescale (tau_g très élevé = forte diffusivité thermique)
omega_f, omega_g = 1./tau_f, 1./tau_g



#======================#
# Numerical parameters #
#======================#
nx = 100
ny = 100
nt = 4000 # nb of time step
dt, dx = 1, 1

rho_0, T_0 = 1, 2


# BC params
u_left = 0.02
T_bottom = 2.2  # légèrement > T0

beta = 1e-3


c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]]) # directional vector
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weighted directions
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


#----------- Initialisation ----------------#
X,Y = np.meshgrid(range(nx), range(ny))

ux = np.zeros((nx, ny))
uy = np.zeros((nx, ny))
rho = np.ones((nx, ny)) * rho_0
T = np.ones((nx, ny)) * T_0
F = np.ones((nx, ny))

#---------------- Distribution -----------------#
f = np.zeros((9, nx, ny))
g = np.zeros((9, nx, ny))

#------------- Create equilibrium state ----------------#
for i in range(9):
    f_eq = w[i] * rho * (1 + 
                         3*(c[i,0]*ux + c[i,1]*uy) + 
                         4.5*(c[i,0]*ux + c[i,1]*uy)**2 - 
                         1.5*(ux**2 + uy**2))
    f[i] = f_eq

    g_eq = w[i] * T * (1 + 3*(c[i,0]*ux + c[i,1]*uy))
    g[i] = g_eq


#----------------- Configuration visualization --------------------#
# Create figure with subplots for real-time visualization
plt.ion()  # Enable interactive mode
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Simulation LBM - Évolution en temps réel')

# Initialize plots
im_T = axes[0].imshow(T, cmap='hot', origin='lower')
axes[0].set_title('Température (T)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
cbar_T = plt.colorbar(im_T, ax=axes[0])

# Velocity magnitude
U_mag = np.sqrt(ux**2 + uy**2)
im_U = axes[1].imshow(U_mag, cmap='viridis', origin='lower')
axes[1].set_title('Magnitude de vélocité')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
cbar_U = plt.colorbar(im_U, ax=axes[1])

plt.tight_layout()
plt.show()


#----------------- Temporal loop --------------------#
for step in range(nt):

    # Buyancy force computation
    Fx = np.zeros((nx, ny))
    Fy = rho * beta * (T - T_0) * gravity

    # Collision (for f)
    for i in range(9):

        cu = 3*(c[i,0]*ux + c[i,1]*uy)  
        f_eq = w[i] * rho * (1 + cu + 0.5*cu**2 - 1.5*(ux**2 + uy**2))

        # Force term
        cu = c[i,0]*ux + c[i,1]*uy

        F = (1 - 0.5*omega_f) * w[i] * (
            ((c[i,0] - ux)/c_s2 + cu*c[i,0]/c_s4) * Fx +
            ((c[i,1] - uy)/c_s2 + cu*c[i,1]/c_s4) * Fy
        )

        f[i] = f[i] - omega_f*(f[i] - f_eq) + F

    # Collision (for g)
    for i in range(9):
        g_eq = w[i] * T * (1 + 3*(c[i,0]*ux + c[i,1]*uy))
        g[i] = g[i] - omega_g*(g[i] - g_eq)
    

    # Streaming (both f and g)
    for i in range(9):
        f[i] = np.roll(f[i], shift=(c[i,0], c[i,1]), axis=(0,1))
        g[i] = np.roll(g[i], shift=(c[i,0], c[i,1]), axis=(0,1))


    # Apply BCs

    #============== LEFT SIDE ===============#
    # Imposed inlet velocity
    rho_left = (f[0,0,:] + f[2,0,:] + f[4,0,:] +
                2*(f[3,0,:] + f[6,0,:] + f[7,0,:])) / (1 - u_left)

    # Zou-He
    f[1,0,:] = f[3,0,:] + 2/3 * rho_left * u_left
    f[5,0,:] = f[7,0,:] + 0.5*(f[4,0,:] - f[2,0,:]) + 1/6 * rho_left * u_left
    f[8,0,:] = f[6,0,:] + 0.5*(f[2,0,:] - f[4,0,:]) + 1/6 * rho_left * u_left

    #========= RIGHT SIDE ==========#
    f[3,-1,:] = f[3,-2,:]
    f[6,-1,:] = f[6,-2,:]
    f[7,-1,:] = f[7,-2,:]

    #========= BOTTOM SIDE ===========#
    # Impermeable (bounce back)
    f[2,:,0] = f[4,:,0]
    f[5,:,0] = f[7,:,0]
    f[6,:,0] = f[8,:,0]


    g[2,:,0] = -g[4,:,0] + 2*w[2]*T_bottom
    g[5,:,0] = -g[7,:,0] + 2*w[5]*T_bottom
    g[6,:,0] = -g[8,:,0] + 2*w[6]*T_bottom


    #========= TOP SIDE ===========#
    # Impermeable (bounce back)
    f[4,:,-1] = f[2,:,-1]
    f[7,:,-1] = f[5,:,-1]
    f[8,:,-1] = f[6,:,-1]

    g[4,:,-1] = g[2,:,-1]
    g[7,:,-1] = g[5,:,-1]
    g[8,:,-1] = g[6,:,-1]


    # Obstacle
    mask = np.zeros((nx, ny), dtype=bool)
    mask[40:60, 40:60] = True
    for i in range(9):
        f[i,mask] = f[opp[i],mask]
        g[i,mask] = g[opp[i],mask]


    # Updata macroscopic variables
    rho = np.sum(f, axis=0)

    c_x = c[:, 0].reshape(9, 1, 1) # to match dim of f = [9, nx, ny]
    ux = (np.sum(c_x*f, axis=0) + 0.5*Fx) / rho

    c_y = c[:,1].reshape(9, 1, 1)
    uy = (np.sum(c_y*f, axis=0) + 0.5*Fy) / rho
    T = np.sum(g, axis=0)

    
    #----------------- Real-time visualization update --------------------#
    # Update every 10 steps to improve performance
    if step % 10 == 0:
        # Update temperature plot
        im_T.set_data(T)
        im_T.set_clim(vmin=T.min(), vmax=T.max())
        axes[0].set_title(f'Température (T) - Étape {step}/{nt}')
        
        # Update velocity magnitude plot
        U_mag = np.sqrt(ux**2 + uy**2)
        im_U.set_data(U_mag)
        im_U.set_clim(vmin=U_mag.min(), vmax=U_mag.max())
        axes[1].set_title(f'Magnitude de vélocité - Étape {step}/{nt}')
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Optional: Print progress
        print(f'Étape {step}/{nt} - T_mean: {T.mean():.3f}, U_max: {U_mag.max():.6f}')


