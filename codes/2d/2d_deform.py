from firedrake import *
from netgen.occ import *

import math
import os
import time

import imageio.v2 as imageio
import matplotlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def env_float(name, default): return float(os.environ.get(name, default))


def env_int(name, default): return int(os.environ.get(name, default))


def env_bool(name, default):
    raw = os.environ.get(name)
    return bool(default) if raw is None else raw.lower() not in {"0", "false", "no", "off"}


radius = env_float("BALL_RADIUS", 0.20)
cx = env_float("BALL_CENTER_X", 0.0)
ground_y_value = env_float("GROUND_Y", 0.0)
initial_gap = env_float("INITIAL_GAP", 0.10)
cy = ground_y_value + radius + initial_gap

mesh_degree = env_int("MESH_DEGREE", 2)
fe_degree = env_int("FE_DEGREE", 1)
maxh = env_float("MESH_MAXH", 0.025)
quad = {"quadrature_degree": env_int("QUADRATURE_DEGREE", 3)}

rho = Constant(env_float("DENSITY", 1.0))
young_modulus_value = env_float("YOUNG_MODULUS", 2.0e3)
E = Constant(young_modulus_value)
nu = Constant(env_float("POISSON_RATIO", 0.30))
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
body_force = Constant((0.0, env_float("GRAVITY", -9.81)))
kv_tau = Constant(env_float("KV_TAU", 1.0e-3))

dt_value = env_float("SIM_DT", 1.5e-4)
num_steps = env_int("SIM_STEPS", 1800)
save_every = env_int("SAVE_EVERY", 9)
dt = Constant(dt_value)

gamma_value = env_float("NEWMARK_GAMMA", 0.5)
beta_value = env_float("NEWMARK_BETA", 0.25 * (gamma_value + 0.5) ** 2)
beta_nm = Constant(beta_value)
gamma_nm = Constant(gamma_value)
initial_velocity = Constant((0.0, env_float("INITIAL_VY", -0.85)))

lvpp_alpha0 = env_float("LVPP_ALPHA0", 2.0e-2)
lvpp_alpha_growth = env_float("LVPP_ALPHA_GROWTH", 1.15)
lvpp_alpha_max = env_float("LVPP_ALPHA_MAX", 8.0e-2)
lvpp_tol = env_float("LVPP_TOL", 1.0e-7)
if lvpp_tol <= 0.0:
    raise ValueError("LVPP_TOL must be positive.")
gap_eq_tol = env_float("GAP_EQ_TOL", 1.0e-10)
penetration_tol = env_float("PENETRATION_TOL", 1.0e-12)
if penetration_tol < 0.0:
    raise ValueError("PENETRATION_TOL must be nonnegative.")
min_gap_tol = env_float("MIN_GAP_TOL", -penetration_tol)
gap_floor = env_float("LVPP_GAP_FLOOR", 1.0e-14)
lvpp_max_it = env_int("LVPP_MAX_IT", 100)
activation_gap = env_float("LVPP_ACTIVATION_GAP", 8.0e-3)

newton_tol = env_float("NEWTON_TOL", 1.0e-12)
newton_max_it = env_int("NEWTON_MAX_IT", 35)
newton_line_search_steps = env_int("NEWTON_LINE_SEARCH_STEPS", 10)

video_name = os.environ.get("SIM_VIDEO_NAME", "2d_deform.mp4")
write_video = env_bool("WRITE_VIDEO", True)
fps = env_int("VIDEO_FPS", 24)

ground_x0 = env_float("GROUND_X0", cx - 1.65 * radius)
ground_x1 = env_float("GROUND_X1", cx + 1.65 * radius)
ground_depth = env_float("GROUND_DEPTH", 0.24)
ground_nx = env_int("GROUND_NX", 64)
ground_ny = env_int("GROUND_NY", 16)
rho_ground = Constant(env_float("GROUND_DENSITY", 1.0))
E_ground_value = env_float("GROUND_YOUNG_MODULUS", 3.0e3)
E_ground = Constant(E_ground_value)
nu_ground = Constant(env_float("GROUND_POISSON_RATIO", 0.30))
mu_ground = E_ground / (2.0 * (1.0 + nu_ground))
lmbda_ground = E_ground * nu_ground / (
    (1.0 + nu_ground) * (1.0 - 2.0 * nu_ground)
)
ground_body_force = Constant((0.0, env_float("GROUND_GRAVITY", 0.0)))
ground_kv_tau = Constant(env_float("GROUND_KV_TAU", 2.0e-3))

ground_length = ground_x1 - ground_x0
ground_dx = ground_length / ground_nx
normal_sample_h = env_float("GROUND_NORMAL_SAMPLE_H", 0.5 * ground_dx)
surface_eps = 1.0e-10


def eps(w):
    return sym(grad(w))


def sigma(w):
    return 2.0 * mu * eps(w) + lmbda * tr(eps(w)) * Identity(2)


def sigma_ground(w):
    return 2.0 * mu_ground * eps(w) + lmbda_ground * tr(eps(w)) * Identity(2)


def flat(func): return np.asarray(func.dat.data_ro, dtype=float).reshape(-1)


def set_flat(func, values):
    func.dat.data[:] = np.asarray(values, dtype=float).reshape(func.dat.data.shape)


def petsc_to_csr(mat):
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


def solve_columns(solver, matrix):
    rhs = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    if rhs.size == 0:
        return np.zeros((matrix.shape[0], 0))
    return np.column_stack([solver(rhs[:, j]) for j in range(rhs.shape[1])])


def ordered_vom_interpolation_matrix(mesh_obj, source_space, points, tol=1.0e-8):
    points = np.asarray(points, dtype=float)
    vom = VertexOnlyMesh(mesh_obj, points, missing_points_behaviour="error")
    target_space = VectorFunctionSpace(vom, "DG", 0, dim=2)
    raw = petsc_to_csr(
        assemble(interpolate(TrialFunction(source_space), target_space)).petscmat
    ).tocsr()
    vom_coords = np.asarray(vom.coordinates.dat.data_ro, dtype=float)
    distances, vom_ids = cKDTree(vom_coords).query(points, k=1)
    max_distance = float(np.max(distances)) if len(distances) else 0.0
    if max_distance > tol:
        raise RuntimeError(
            f"VertexOnlyMesh point reorder failed: max distance {max_distance:.3e}"
        )
    rows = np.empty(2 * len(vom_ids), dtype=int)
    rows[0::2] = 2 * vom_ids
    rows[1::2] = 2 * vom_ids + 1
    return raw[rows, :].tocsr()


def normalized_rows(values, fallback):
    values = np.asarray(values, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    if fallback.ndim == 1:
        fallback = np.broadcast_to(fallback, values.shape)
    norms = np.linalg.norm(values, axis=1)
    good = norms > 1.0e-14
    out = np.array(fallback, dtype=float, copy=True)
    out[good] = values[good] / norms[good, None]
    return out, np.maximum(norms, 1.0e-14)


def ordered_boundary_loop(triangles):
    counts = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    edges = [edge for edge, count in counts.items() if count == 1]
    neighbors = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    start = edges[0][0]
    loop, prev, cur = [start], None, start
    while True:
        nxt = next((node for node in neighbors[cur] if node != prev), start)
        loop.append(nxt)
        if nxt == start:
            return np.asarray(loop, dtype=int)
        prev, cur = cur, nxt


def build_disk_geometry():
    disk = WorkPlane(Axes((0.0, 0.0, 0.0), n=Z, h=X)).Circle(1).Face()
    rectangle = WorkPlane().Rectangle(2, 2).Face()
    shape = Glue([disk - rectangle.Move((-1, 0, 0)), disk - rectangle.Move((-1, -2, 0))])
    shape = shape.Scale(Pnt(0, 0, 0), radius).Move((cx, cy, 0.0))
    shape.edges.Min(Y).name = "bottom"
    return OCCGeometry(shape, dim=2)


geo = build_disk_geometry()
ngmesh = geo.GenerateMesh(maxh=maxh)
no_overlap = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}
ball_mesh = Mesh(ngmesh, distribution_parameters=no_overlap)
ball_mesh = Mesh(ball_mesh.curve_field(mesh_degree), distribution_parameters=no_overlap)

contact_labels = [i + 1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bottom"]
if not contact_labels:
    raise RuntimeError("Could not find the Netgen boundary label 'bottom'.")
contact_mesh = Submesh(ball_mesh, 1, contact_labels)

ground_mesh = RectangleMesh(ground_nx, ground_ny, ground_length, ground_depth, distribution_parameters=no_overlap)
ground_mesh.coordinates.dat.data[:, 0] += ground_x0
ground_mesh.coordinates.dat.data[:, 1] += ground_y_value - ground_depth
ground_bottom_id = 3

dx_ball = Measure("dx", ball_mesh, metadata=quad)
dx_ground = Measure("dx", ground_mesh, metadata=quad)
dx_contact = Measure("dx", contact_mesh, metadata=quad)


V = VectorFunctionSpace(ball_mesh, "CG", fe_degree)
Vc = VectorFunctionSpace(contact_mesh, "CG", fe_degree)
Q = FunctionSpace(contact_mesh, "CG", fe_degree)
G = VectorFunctionSpace(ground_mesh, "CG", fe_degree)
W_dim = V.dim() + G.dim() + Q.dim()

u_old, v_old, a_old, u_state, u_prev_state = [
    Function(V, name=name) for name in ("u_old", "v_old", "a_old", "u_state", "u_prev_state")
]
ball_a_new, ball_v_new = [Function(V, name=name) for name in ("ball_a_new", "ball_v_new")]
ground_u_old, ground_v_old, ground_a_old, ground_state, ground_prev_state = [
    Function(G, name=name)
    for name in ("ground_u_old", "ground_v_old", "ground_a_old", "ground_state", "ground_prev_state")
]
ground_a_new, ground_v_new = [Function(G, name=name) for name in ("ground_a_new", "ground_v_new")]

v_old.interpolate(initial_velocity)
a_old.interpolate(body_force)
ground_a_old.interpolate(ground_body_force)

ground_bc = DirichletBC(G, Constant((0.0, 0.0)), ground_bottom_id)


def ball_accel_expr(w):
    return (w - u_old - dt * v_old - dt**2 * (0.5 - beta_nm) * a_old) / (beta_nm * dt**2)


def ball_vel_expr(acc):
    return v_old + dt * ((1.0 - gamma_nm) * a_old + gamma_nm * acc)


def ground_accel_expr(w):
    return (w - ground_u_old - dt * ground_v_old - dt**2 * (0.5 - beta_nm) * ground_a_old) / (beta_nm * dt**2)


def ground_vel_expr(acc):
    return ground_v_old + dt * (
        (1.0 - gamma_nm) * ground_a_old + gamma_nm * acc
    )


# Mechanics matrices are constant because the material model is linear elastic
# and Newmark contributes a linear inertial term.
du = TrialFunction(V)
vb = TestFunction(V)
ball_a_trial = du / (beta_nm * dt**2)
ball_v_trial = gamma_nm * du / (beta_nm * dt)
A_ball_form = (
    rho * inner(ball_a_trial, vb)
    + inner(sigma(du), eps(vb))
    + inner(kv_tau * sigma(ball_v_trial), eps(vb))
) * dx_ball
A_ball = petsc_to_csr(assemble(A_ball_form).petscmat).tocsr()
ball_lu = spla.factorized(A_ball.tocsc())

dg = TrialFunction(G)
wg = TestFunction(G)
ground_a_trial = dg / (beta_nm * dt**2)
ground_v_trial = gamma_nm * dg / (beta_nm * dt)
A_ground_form = (
    rho_ground * inner(ground_a_trial, wg)
    + inner(sigma_ground(dg), eps(wg))
    + inner(ground_kv_tau * sigma_ground(ground_v_trial), eps(wg))
) * dx_ground
A_ground = petsc_to_csr(assemble(A_ground_form, bcs=ground_bc).petscmat).tocsr()
ground_lu = spla.factorized(A_ground.tocsc())


def ball_residual_form():
    a_expr = ball_accel_expr(u_state)
    v_expr = ball_vel_expr(a_expr)
    return (
        rho * inner(a_expr, vb)
        + inner(sigma(u_state), eps(vb))
        + inner(kv_tau * sigma(v_expr), eps(vb))
        - rho * inner(body_force, vb)
    ) * dx_ball


def ground_residual_form():
    a_expr = ground_accel_expr(ground_state)
    v_expr = ground_vel_expr(a_expr)
    return (
        rho_ground * inner(a_expr, wg)
        + inner(sigma_ground(ground_state), eps(wg))
        + inner(ground_kv_tau * sigma_ground(v_expr), eps(wg))
        - rho_ground * inner(ground_body_force, wg)
    ) * dx_ground


R_ball_zero_form = ball_residual_form()
R_ground_zero_form = ground_residual_form()


def zero_ball_residual():
    u_state.dat.data[:] = 0.0
    return np.asarray(assemble(R_ball_zero_form).dat.data_ro, dtype=float).reshape(-1)


def zero_ground_residual():
    ground_state.dat.data[:] = 0.0
    return np.asarray(assemble(R_ground_zero_form, bcs=ground_bc).dat.data_ro, dtype=float).reshape(-1)


def solve_free_ball(rb0): return ball_lu(-rb0)


def solve_free_ground(rg0): return ground_lu(-rg0)


Xc = SpatialCoordinate(contact_mesh)
contact_ref = Function(Vc, name="contact_ref")
contact_ref.interpolate(as_vector((Xc[0], Xc[1])))
contact_coords = np.asarray(contact_ref.dat.data_ro, dtype=float).copy()
num_contact = contact_coords.shape[0]

I_ball_contact = petsc_to_csr(assemble(interpolate(TrialFunction(V), Vc)).petscmat).tocsr()
B_ball_components = tuple(I_ball_contact[i::2, :].tocsr() for i in range(2))

M_contact = petsc_to_csr(assemble(TrialFunction(Q) * TestFunction(Q) * dx_contact).petscmat).tocsr()

contact_x = np.clip(contact_coords[:, 0], ground_x0 + surface_eps, ground_x1 - surface_eps)


def ground_trace_points(offset=0.0):
    x = np.clip(contact_x + offset, ground_x0 + surface_eps, ground_x1 - surface_eps)
    return np.column_stack((x, np.full(num_contact, ground_y_value - surface_eps)))


ground_points = ground_trace_points()
ground_piola_points = {"xm": ground_trace_points(-normal_sample_h), "xp": ground_trace_points(normal_sample_h)}

I_ground_contact = ordered_vom_interpolation_matrix(ground_mesh, G, ground_points)
B_ground_components = tuple(I_ground_contact[i::2, :].tocsr() for i in range(2))
I_ground_normal = {
    name: ordered_vom_interpolation_matrix(ground_mesh, G, points)
    for name, points in ground_piola_points.items()
}
B_ground_normal = {
    name: tuple(I_ground_normal[name][i::2, :].tocsr() for i in range(2))
    for name in ground_piola_points
}

nb = V.dim()
ng = G.dim()
nc = num_contact
psi_lag = np.zeros(nc)


def contact_ball_displacement(ub): return (I_ball_contact @ ub).reshape((-1, 2))


def contact_ground_displacement(ug): return (I_ground_contact @ ug).reshape((-1, 2))


def ground_surface_position(ug, name):
    return ground_piola_points[name] + (I_ground_normal[name] @ ug).reshape((-1, 2))


def deformed_ground_geometry(ug):
    pm = ground_surface_position(ug, "xm")
    pp = ground_surface_position(ug, "xp")
    tangent = pp - pm
    # Piola normal sampled from the deformed ground tangent.
    piola_normal_unnorm = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    raw_normal, tangent_norm = normalized_rows(piola_normal_unnorm, np.asarray((0.0, 1.0)))
    orientation = np.ones(nc)
    flip = raw_normal[:, 1] < 0.0
    orientation[flip] = -1.0
    normal = raw_normal * orientation[:, None]
    return normal, tangent, tangent_norm, orientation


def normal_projected_matrix(component_matrices, normals):
    projected = sp.diags(normals[:, 0], format="csr") @ component_matrices[0]
    projected += sp.diags(normals[:, 1], format="csr") @ component_matrices[1]
    return projected.tocsr()


def contact_geometry(ub, ug):
    normal, tangent, tangent_norm, orientation = deformed_ground_geometry(ug)
    relative = (
        contact_coords
        + contact_ball_displacement(ub)
        - ground_points
        - contact_ground_displacement(ug)
    )
    gap = np.einsum("ij,ij->i", relative, normal)
    return gap, normal, relative, tangent, tangent_norm, orientation


def contact_projection_matrices(normal, relative, tangent, tangent_norm, orientation):
    C_ball = normal_projected_matrix(B_ball_components, normal)
    B_ground_n = normal_projected_matrix(B_ground_components, normal)

    Dtx = B_ground_normal["xp"][0] - B_ground_normal["xm"][0]
    Dty = B_ground_normal["xp"][1] - B_ground_normal["xm"][1]

    safe_s = np.maximum(tangent_norm, 1.0e-14)
    r_dot_n = np.einsum("ij,ij->i", relative, normal)
    rt_r = np.column_stack((relative[:, 1], -relative[:, 0]))
    normal_deriv_weight = (
        orientation[:, None] * rt_r / safe_s[:, None]
        - (r_dot_n / (safe_s * safe_s))[:, None] * tangent
    )
    normal_term = sp.diags(normal_deriv_weight[:, 0], format="csr") @ Dtx
    normal_term += sp.diags(normal_deriv_weight[:, 1], format="csr") @ Dty

    C_ground = (-B_ground_n + normal_term).tocsr()
    J_ball = (C_ball.T @ M_contact).tocsr()
    J_ground = (C_ground.T @ M_contact).tocsr()
    return J_ball, J_ground, C_ball, C_ground


def gap_values(ub, ug): return contact_geometry(ub, ug)[0]


def weak_slack_residual(gap, psi): return M_contact @ (gap - np.exp(psi))


def weak_slack_residual_error(gap, psi):
    return float(np.linalg.norm(weak_slack_residual(gap, psi)) / max(math.sqrt(nc), 1.0))


def initialise_lvpp_lag(ub, ug):
    global psi_lag
    psi_lag = np.log(np.maximum(gap_values(ub, ug), gap_floor))


def contact_residual(ub, ug, psi, rb0, rg0, alpha_value, lag):
    gap, normal, relative, tangent, tangent_norm, orientation = contact_geometry(ub, ug)
    J_ball, J_ground, _, _ = contact_projection_matrices(
        normal,
        relative,
        tangent,
        tangent_norm,
        orientation,
    )
    dpsi = psi - lag
    rb = alpha_value * (A_ball @ ub + rb0) + J_ball @ dpsi
    rg = alpha_value * (A_ground @ ug + rg0) + J_ground @ dpsi
    # Weak LVPP Signorini relation: (g_phys - exp(psi), w)_Gamma = 0.
    rc = weak_slack_residual(gap, psi)
    residual = np.concatenate((rb, rg, rc))
    return residual, gap, (normal, relative, tangent, tangent_norm, orientation)


def contact_newton_step(residual, psi, geometry, alpha_value):
    rb = residual[:nb]
    rg = residual[nb : nb + ng]
    rc = residual[nb + ng :]
    normal, relative, tangent, tangent_norm, orientation = geometry
    J_ball, J_ground, C_ball, C_ground = contact_projection_matrices(
        normal,
        relative,
        tangent,
        tangent_norm,
        orientation,
    )

    base_ball = ball_lu(-rb / alpha_value)
    base_ground = ground_lu(-rg / alpha_value)
    Ainv_J_ball = solve_columns(ball_lu, J_ball) / alpha_value
    Ainv_J_ground = solve_columns(ground_lu, J_ground) / alpha_value

    C_ball_weak = (M_contact @ C_ball).tocsr()
    C_ground_weak = (M_contact @ C_ground).tocsr()
    Dpsi = -(M_contact @ sp.diags(np.exp(psi), format="csr")).toarray()
    Schur = Dpsi
    Schur -= C_ball_weak @ Ainv_J_ball
    Schur -= C_ground_weak @ Ainv_J_ground
    rhs_psi = -rc - C_ball_weak @ base_ball - C_ground_weak @ base_ground

    try:
        dpsi = np.linalg.solve(Schur, rhs_psi)
    except np.linalg.LinAlgError:
        dpsi = np.linalg.lstsq(Schur, rhs_psi, rcond=None)[0]

    dub = base_ball - Ainv_J_ball @ dpsi
    dug = base_ground - Ainv_J_ground @ dpsi
    return np.concatenate((dub, dug, dpsi))


def split_unknown(x):
    return x[:nb], x[nb : nb + ng], x[nb + ng :]


def saddle_newton_solve(x0, rb0, rg0, alpha_value, lag):
    x = x0.copy()
    best_norm = np.inf
    last = None
    for it in range(newton_max_it):
        ub, ug, psi = split_unknown(x)
        residual, gap, geometry = contact_residual(
            ub,
            ug,
            psi,
            rb0,
            rg0,
            alpha_value,
            lag,
        )
        norm = float(np.linalg.norm(residual) / max(math.sqrt(residual.size), 1.0))
        gap_eq = weak_slack_residual_error(gap, psi)
        last = (it + 1, norm, gap_eq, float(np.min(gap)))
        if norm < newton_tol and gap_eq < max(newton_tol, 0.25 * gap_eq_tol):
            return x, last

        best_norm = min(best_norm, norm)
        dx = contact_newton_step(residual, psi, geometry, alpha_value)
        damping = 1.0
        accepted = False
        for _ in range(newton_line_search_steps):
            trial = x + damping * dx
            tub, tug, tpsi = split_unknown(trial)
            trial_residual, trial_gap, _ = contact_residual(
                tub,
                tug,
                tpsi,
                rb0,
                rg0,
                alpha_value,
                lag,
            )
            trial_norm = float(
                np.linalg.norm(trial_residual)
                / max(math.sqrt(trial_residual.size), 1.0)
            )
            trial_gap_eq = weak_slack_residual_error(trial_gap, tpsi)
            if trial_norm <= 1.05 * norm or trial_norm <= best_norm:
                x = trial
                accepted = True
                break
            if trial_gap_eq < gap_eq and trial_norm < 5.0 * norm:
                x = trial
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            x = x + dx

    return x, last


def lvpp_alpha(k): return min(lvpp_alpha0 * lvpp_alpha_growth**k, lvpp_alpha_max)


def lvpp_increment(ub, ug, ub_prev, ug_prev):
    dub, dug = ub - ub_prev, ug - ug_prev
    return math.sqrt(float(np.mean(dub * dub) + np.mean(dug * dug)))


def monolithic_lvpp_solve(rb0, rg0, ub_initial, ug_initial):
    global psi_lag
    initialise_lvpp_lag(flat(u_old), flat(ground_u_old))
    psi_initial = psi_lag.copy()
    x = np.concatenate((ub_initial, ug_initial, psi_initial))
    last = None

    for k in range(lvpp_max_it):
        alpha_value = lvpp_alpha(k)
        lag = psi_lag.copy()
        ub_prev, ug_prev, _ = split_unknown(x)
        x, newton_info = saddle_newton_solve(x, rb0, rg0, alpha_value, lag)
        ub, ug, psi = split_unknown(x)
        gap = gap_values(ub, ug)
        gap_eq = weak_slack_residual_error(gap, psi)
        du = lvpp_increment(ub, ug, ub_prev, ug_prev)
        pressure = np.maximum(0.0, -(psi - lag) / alpha_value)
        active = int(np.count_nonzero(pressure > 1.0e-9))
        max_pressure = float(np.max(pressure)) if pressure.size else 0.0
        min_gap = float(np.min(gap))
        last = (k + 1, alpha_value, du, gap_eq, min_gap, active, max_pressure, newton_info)

        if du < lvpp_tol and gap_eq < gap_eq_tol and min_gap > min_gap_tol:
            psi_lag = psi.copy()
            return x, last

        psi_lag = psi.copy()

    k, alpha_value, du, gap_eq, min_gap, active, max_pressure, newton_info = last
    raise RuntimeError(
        "LVPP monolithic step did not converge: "
        f"it={k}, du={du:.3e}, gap_eq={gap_eq:.3e}, "
        f"min_gap={min_gap:.3e}, min_gap_tol={min_gap_tol:.3e}, "
        f"penetration_tol={penetration_tol:.3e}, "
        f"alpha={alpha_value:.3e}, "
        f"newton={newton_info}"
    )


def accept_ball(ub):
    set_flat(u_state, ub)
    ball_a_new.interpolate(ball_accel_expr(u_state))
    ball_v_new.interpolate(ball_vel_expr(ball_a_new))
    u_old.assign(u_state)
    v_old.assign(ball_v_new)
    a_old.assign(ball_a_new)


def accept_ground(ug):
    set_flat(ground_state, ug)
    ground_a_new.interpolate(ground_accel_expr(ground_state))
    ground_v_new.interpolate(ground_vel_expr(ground_a_new))
    ground_u_old.assign(ground_state)
    ground_v_old.assign(ground_v_new)
    ground_a_old.assign(ground_a_new)


def current_min_gap(): return float(np.min(gap_values(flat(u_old), flat(ground_u_old))))


frames = []
P1_ball = FunctionSpace(ball_mesh, "CG", 1)
V_ball_render = VectorFunctionSpace(ball_mesh, "CG", 1)
X_ball = SpatialCoordinate(ball_mesh)
x_ball_render = Function(V_ball_render, name="x_ball_render")
u_ball_render = Function(V_ball_render, name="u_ball_render")
x_ball_render.interpolate(as_vector((X_ball[0], X_ball[1])))
ball_coords0 = np.asarray(x_ball_render.dat.data_ro, dtype=float).copy()
ball_triangles = np.asarray(P1_ball.cell_node_map().values, dtype=int)
ball_boundary_loop = ordered_boundary_loop(ball_triangles)

P1_ground = FunctionSpace(ground_mesh, "CG", 1)
V_ground_render = VectorFunctionSpace(ground_mesh, "CG", 1)
X_ground = SpatialCoordinate(ground_mesh)
x_ground_render = Function(V_ground_render, name="x_ground_render")
u_ground_render = Function(V_ground_render, name="u_ground_render")
x_ground_render.interpolate(as_vector((X_ground[0], X_ground[1])))
ground_coords0 = np.asarray(x_ground_render.dat.data_ro, dtype=float).copy()
ground_triangles = np.asarray(P1_ground.cell_node_map().values, dtype=int)
ground_top_nodes = np.flatnonzero(np.isclose(ground_coords0[:, 1], ground_y_value, atol=1.0e-12))


def add_mesh(ax, coords, triangles, facecolor, edgecolor, linewidth):
    ax.add_collection(
        PolyCollection(
            coords[triangles],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            antialiased=True,
        )
    )


def render_frame():
    if not write_video:
        return

    u_ball_render.interpolate(u_old)
    ball_coords = ball_coords0 + np.asarray(u_ball_render.dat.data_ro, dtype=float)

    u_ground_render.interpolate(ground_u_old)
    ground_coords = ground_coords0 + np.asarray(u_ground_render.dat.data_ro, dtype=float)

    top = ground_coords[ground_top_nodes]
    top = top[np.argsort(top[:, 0])]

    fig, ax = plt.subplots(figsize=(5, 7), dpi=100)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    add_mesh(ax, ground_coords, ground_triangles, (0.68, 0.68, 0.64, 0.35), (0.18, 0.18, 0.18, 0.12), 0.24)
    add_mesh(ax, ball_coords, ball_triangles, (0.80, 0.90, 0.96, 0.36), (0.50, 0.70, 0.84, 0.18), 0.26)
    ax.plot(
        ball_coords[ball_boundary_loop, 0],
        ball_coords[ball_boundary_loop, 1],
        color=(0.19, 0.49, 0.70, 0.95),
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        top[:, 0],
        top[:, 1],
        color=(0.0, 0.0, 0.0, 0.95),
        linewidth=2.2,
        solid_capstyle="round",
    )

    ax.set_aspect("equal", adjustable="box")
    x_margin = 0.45 * radius
    ax.set_xlim(ground_x0 - x_margin, ground_x1 + x_margin)
    y_lower = ground_y_value - ground_depth - 0.05 * radius
    ax.set_ylim(y_lower, cy + 1.05 * radius)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)


print(f"2d_deform.py | Vb={V.dim()} Vg={G.dim()} Q={Q.dim()} total={W_dim} | dt={dt_value:.2e} steps={num_steps} | Eb={young_modulus_value:.2e} Eg={E_ground_value:.2e} | lvpp_tol={lvpp_tol:.1e} min_gap_tol={min_gap_tol:.1e} | normal=Piola | gap=weak slack | solver=Schur Newton")

simulation_start = time.perf_counter()
contact_steps = 0
max_lvpp_iters = 0
global_min_gap = current_min_gap()
log_every = env_int("LOG_EVERY", 50)

render_frame()

for step in range(1, num_steps + 1):
    rb0 = zero_ball_residual()
    rg0 = zero_ground_residual()
    ub_free = solve_free_ball(rb0)
    ug_free = solve_free_ground(rg0)
    trial_gap = float(np.min(gap_values(ub_free, ug_free)))
    lvpp_info = None

    if trial_gap > activation_gap:
        accept_ball(ub_free)
        accept_ground(ug_free)
        min_gap = trial_gap
    else:
        contact_steps += 1
        x, lvpp_info = monolithic_lvpp_solve(rb0, rg0, ub_free, ug_free)
        ub, ug, psi = split_unknown(x)
        accept_ball(ub)
        accept_ground(ug)
        min_gap = float(np.min(gap_values(ub, ug)))
        max_lvpp_iters = max(max_lvpp_iters, lvpp_info[0])

    global_min_gap = min(global_min_gap, min_gap)

    if step % save_every == 0 or step == num_steps:
        render_frame()

    should_log = step % log_every == 0 or (lvpp_info is not None and contact_steps == 1)
    if should_log:
        if lvpp_info is None:
            print(f"step {step:04d} | free    | trial_gap={trial_gap:.3e}")
        else:
            (lvpp_it, alpha_value, du_value, gap_eq, _, active, max_pressure, newton_info) = lvpp_info
            print(
                f"step {step:04d} | contact | min_gap={min_gap:.3e} | "
                f"lvpp={lvpp_it} | alpha={alpha_value:.2e} | du={du_value:.3e} | "
                f"gap_eq={gap_eq:.3e} | active={active}/{nc} | "
                f"pmax={max_pressure:.3e} | newton={newton_info[0]}"
            )

if write_video and frames:
    imageio.mimsave(
        video_name,
        frames + [frames[-1]] * env_int("VIDEO_PAUSE_FRAMES", 16),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    print(f"MP4 written: {os.path.abspath(video_name)}")

print(f"simulation complete | contact_steps={contact_steps} | max_lvpp_iters={max_lvpp_iters} | global_min_gap={global_min_gap:.3e} | wall={time.perf_counter() - simulation_start:.2f}s")
