import taichi as ti
import taichi.math as tm


@ti.kernel
def compute_min_max(T: ti.template()) -> ti.math.vec2:
    tmin = 1e9
    tmax = -1e9
    for i, j in T:
        val = T[i, j]
        if val < tmin:
            tmin = val
        if val > tmax:
            tmax = val
    return ti.math.vec2(tmin, tmax)

import taichi as ti

@ti.func
def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b

@ti.func
def viridis(t: float) -> ti.math.vec3:
    
    t = ti.math.clamp(t, 0.0, 1.0)
    r, g, b = 0.0, 0.0, 0.0

    if t < 0.25:
        t_local = t / 0.25
        r = _lerp(0.267, 0.231, t_local)
        g = _lerp(0.004, 0.322, t_local)
        b = _lerp(0.329, 0.545, t_local)
    elif t < 0.5:
        t_local = (t - 0.25) / 0.25
        r = _lerp(0.231, 0.129, t_local)
        g = _lerp(0.322, 0.569, t_local)
        b = _lerp(0.545, 0.549, t_local)
    elif t < 0.75:
        t_local = (t - 0.5) / 0.25
        r = _lerp(0.129, 0.369, t_local)
        g = _lerp(0.569, 0.788, t_local)
        b = _lerp(0.549, 0.384, t_local)
    else:
        t_local = (t - 0.75) / 0.25
        r = _lerp(0.369, 0.992, t_local)
        g = _lerp(0.788, 0.906, t_local)
        b = _lerp(0.384, 0.145, t_local)

    return ti.math.vec3(r, g, b)

@ti.kernel
def fill_image(image_field: ti.template(), T: ti.template(),
               T_min: float, T_max: float, scale: int):
    for i, j in image_field:
        src_i = i // scale
        src_j = j // scale
        val = (T[src_i, src_j] - T_min) / (T_max - T_min + 1e-8)
        val = ti.math.clamp(val, 0.0, 1.0)
        image_field[i, j] = viridis(val)



@ti.kernel
def compute_velocity_arrows(ux: ti.template(), uy: ti.template(),
                            nx: ti.i32, ny: ti.i32,
                            scale: int,
                            arrow_origins: ti.template(),
                            arrow_directions: ti.template(),
                            arrow_count: ti.template()):
    idx = 0
    for i, j in ti.ndrange(nx, ny):
        if i % scale == 0 and j % scale == 0:
            if idx < arrow_origins.shape[0]:
                # Position normalisée [0,1] (origine en bas à gauche pour ti.GUI)
                norm_x = i / (nx - 1)
                norm_y = j / (ny - 1)
                arrow_origins[idx] = tm.vec2(norm_x, norm_y)

                vx = ux[i, j]
                vy = uy[i, j]
                vmag = ti.sqrt(vx * vx + vy * vy)

                vx_norm = 0.0
                vy_norm = 0.0
                if vmag > 1e-6:
                    vx_norm = vx / vmag * 0.03
                    vy_norm = vy / vmag * 0.03   # pas de signe négatif (Y vers le haut)

                arrow_directions[idx] = tm.vec2(vx_norm, vy_norm)
                idx += 1
    arrow_count[None] = idx