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

@ti.kernel
def fill_image(image_field: ti.template(), T: ti.template(), T_min: float, T_max: float, scale: int):
    for i, j in image_field:
        src_i = i // scale
        src_j = j // scale
        val = (T[src_i, src_j] - T_min) / (T_max - T_min + 1e-8)
        val = tm.clamp(val, 0.0, 1.0)  # sécurité
        r, g, b = 1.0, 1.0, 1.0
        if val < 0.33:
            r = 0.0
            g = val / 0.33
            b = 1.0
        elif val < 0.66:
            r = (val - 0.33) / 0.33
            g = 1.0
            b = 1.0 - (val - 0.33) / 0.33
        else:
            r = 1.0
            g = 1.0 - (val - 0.66) / 0.34
            b = 0.0
        image_field[i, j] = tm.vec3(r, g, b)
