"""
3D rigid ground case.

This is the 3D elastic ball against a fixed plane. The ground has no deformable
mesh; it only appears in the gap as the plane y = ground_y. The time step is the
same pattern as the 2D case: free-flight first, then LVPP/Newton if contact starts.
"""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Build the 3D ball mesh and extract a bottom-cap contact mesh.
def build_bottom_cap_sphere(radius, center, contact_threshold, maxh, mesh_degree, contact_maxh=None):
    cx, cy, cz = center
    ball = Sphere(Pnt(0.0, 0.0, 0.0), radius)
    y_cut = contact_threshold * radius
    cut_height = 4.0 * radius
    box_half_width = 1.5 * radius

    # Keep only the part of the sphere that lies inside this bounding box.
    def cut_box(y0, y1):
        return Box(
            Pnt(-box_half_width, y0, -box_half_width),
            Pnt(box_half_width, y1, box_half_width),
        )

    bottom_patch = ball - cut_box(y_cut, y_cut + cut_height)
    upper_part = ball - cut_box(y_cut - cut_height, y_cut)
    bottom_patch.faces.Min(Y).name = "bottom"
    if contact_maxh is not None:
        bottom_patch.faces.Min(Y).maxh = contact_maxh
    shape = Glue([upper_part, bottom_patch]).Move((cx, cy, cz))
    shape.faces.Min(Y).name = "bottom"
    if contact_maxh is not None:
        shape.faces.Min(Y).maxh = contact_maxh

    ngmesh = OCCGeometry(shape, dim=3).GenerateMesh(maxh=maxh)
    no_overlap = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}
    mesh = Mesh(Mesh(ngmesh).curve_field(mesh_degree), distribution_parameters=no_overlap)
    labels = [
        i + 1
        for i, name in enumerate(ngmesh.GetRegionNames(codim=1))
        if name == "bottom"
    ]
    if not labels:
        raise RuntimeError("Could not find the Netgen boundary label 'bottom'.")
    contact_mesh = Submesh(mesh, 2, labels[0])
    return mesh, contact_mesh, labels


# Extract exterior surface triangles so Matplotlib can draw the 3D mesh.
def make_mesh_surface_triangles(mesh):
    cell_nodes = np.asarray(mesh.coordinates.cell_node_map().values, dtype=int)
    exterior_cells = np.asarray(mesh.exterior_facets.facet_cell, dtype=int)[:, 0]
    exterior_facets = np.asarray(
        mesh.exterior_facets.local_facet_dat.data_ro,
        dtype=int,
    )

    coord_element = mesh.coordinates.function_space().finat_element
    entity_dofs = coord_element.entity_dofs()
    closure_dofs = coord_element.entity_closure_dofs()
    vertex_dofs = {
        dofs[0]
        for dofs in entity_dofs[0].values()
        if len(dofs) == 1
    }

    edge_midpoint = {}
    for dofs in closure_dofs[1].values():
        vertices = tuple(sorted(dof for dof in dofs if dof in vertex_dofs))
        mids = [dof for dof in dofs if dof not in vertex_dofs]
        if len(vertices) == 2 and len(mids) == 1:
            edge_midpoint[vertices] = mids[0]

    # Turn one Firedrake exterior facet into one or more renderable triangles.
    def split_facet(local_facet):
        local_dofs = list(closure_dofs[2][int(local_facet)])
        vertices = [dof for dof in local_dofs if dof in vertex_dofs]
        if len(vertices) != 3:
            return [tuple(local_dofs[:3])]

        v0, v1, v2 = vertices
        m01 = edge_midpoint.get(tuple(sorted((v0, v1))))
        m12 = edge_midpoint.get(tuple(sorted((v1, v2))))
        m20 = edge_midpoint.get(tuple(sorted((v2, v0))))
        if None in (m01, m12, m20):
            return [(v0, v1, v2)]
        return [
            (v0, m01, m20),
            (m01, v1, m12),
            (m20, m12, v2),
            (m01, m12, m20),
        ]

    triangles = []
    for cell, local_facet in zip(exterior_cells, exterior_facets):
        for tri in split_facet(local_facet):
            triangles.append([cell_nodes[cell, dof] for dof in tri])
    return np.asarray(triangles, dtype=int)



# parameters

# Parameters are plain values here. Edit these lines directly for a different run.

radius = 0.20
cx = 0.0
cz = 0.0
ground_y = 0.0
initial_gap = 0.45
cy = ground_y + radius + initial_gap

mesh_degree = 2
fe_degree = 2
maxh = 0.08
contact_maxh = 0.035
contact_threshold = -0.75
quad = {"quadrature_degree": 3}

rho = Constant(1.0)
young_modulus_value = 5.0e3
E = Constant(young_modulus_value)
nu = Constant(0.30)
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
body_force = Constant((0.0, -9.81, 0.0))
kv_tau_value = 0.0
kv_tau = Constant(kv_tau_value)

dt_value = 2.0e-4
num_steps = 1600
save_every = 30
dt = Constant(dt_value)

gamma_value = 0.5
beta_value = 0.25 * (gamma_value + 0.5) ** 2
gamma_nm = Constant(gamma_value)
beta_nm = Constant(beta_value)
initial_velocity = Constant((0.0, -1.0, 0.0))

lvpp_alpha0 = 2.0e-3
lvpp_alpha_growth = 1.15
lvpp_alpha_max = 2.0e-2
lvpp_tol = 1.0e-6
if lvpp_tol <= 0.0:
    raise ValueError("LVPP_TOL must be positive.")
gap_eq_tol = 1.0e-8
penetration_tol = 1.0e-10
if penetration_tol < 0.0:
    raise ValueError("PENETRATION_TOL must be nonnegative.")
min_gap_tol = -penetration_tol
gap_floor = 1.0e-12
activation_gap = 8.0e-3
entry_gap_default = min(
    0.1 * activation_gap,
    max(10.0 * gap_floor, 10.0 * penetration_tol, 1.0e-14),
)
entry_gap_target = entry_gap_default
entry_backtrack_iters = 40
entry_theta_safety = 0.999
if entry_gap_target <= 0.0:
    raise ValueError("LVPP_ENTRY_GAP_TARGET must be positive.")
if entry_gap_target >= activation_gap:
    raise ValueError("LVPP_ENTRY_GAP_TARGET should be smaller than LVPP_ACTIVATION_GAP.")
if entry_backtrack_iters < 1:
    raise ValueError("LVPP_ENTRY_BACKTRACK_ITERS must be positive.")
if not (0.0 < entry_theta_safety <= 1.0):
    raise ValueError("LVPP_ENTRY_THETA_SAFETY must be in (0, 1].")
lvpp_max_it = 80
require_gap_equation = True
newton_tol = 1.0e-10
newton_max_it = 35
newton_line_search_steps = 12

video_name = "3d_rigid.mp4"
write_video = True
fps = 24

ground_x0 = cx - 1.65 * radius
ground_x1 = cx + 1.65 * radius
ground_z0 = cz - 1.65 * radius
ground_z1 = cz + 1.65 * radius


# helpers

# Mostly conversions between Firedrake Functions and NumPy/SciPy vectors.

# Read a Firedrake Function as a one-dimensional NumPy array.
def flat(func):
    return np.asarray(func.dat.data_ro, dtype=float).reshape(-1)


# Write a one-dimensional NumPy array back into a Firedrake Function.
def set_flat(func, values):
    func.dat.data[:] = np.asarray(values, dtype=float).reshape(func.dat.data.shape)


# Convert a PETSc matrix into SciPy CSR format for the custom linear algebra.
def petsc_to_csr(mat):
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


# Solve the same sparse linear system for several right-hand sides.
def solve_columns(solver, matrix):
    rhs = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    if rhs.size == 0:
        return np.zeros((matrix.shape[0], 0))
    return np.column_stack([solver(rhs[:, j]) for j in range(rhs.shape[1])])


# mesh and spaces

# The ball is a 3D volume mesh. The contact mesh is just the lower cap where gap/psi live.

mesh, contact_mesh, contact_labels = build_bottom_cap_sphere(
    radius,
    (cx, cy, cz),
    contact_threshold,
    maxh,
    mesh_degree,
    contact_maxh,
)

dx_main = Measure(
    "dx",
    mesh,
    metadata=quad,
    intersect_measures=(Measure("dx", mesh), Measure("ds", contact_mesh)),
)
ds_contact = Measure(
    "dx",
    contact_mesh,
    metadata=quad,
    intersect_measures=(Measure("ds", mesh),),
)

V = VectorFunctionSpace(mesh, "CG", fe_degree)
contact_space_family = "CG"
contact_space_degree = 1
Q = FunctionSpace(
    contact_mesh,
    contact_space_family,
    contact_space_degree,
)

u_old = Function(V, name="u_old")
v_old = Function(V, name="v_old")
a_old = Function(V, name="a_old")
u_state = Function(V, name="u_state")
u_entry = Function(V, name="u_entry")
v_new = Function(V, name="v_new")
a_new = Function(V, name="a_new")

V_contact = VectorFunctionSpace(contact_mesh, contact_space_family, contact_space_degree)
u_contact = Function(V_contact, name="u_contact")
x_contact_ref = Function(V_contact, name="x_contact_ref")
X_contact = SpatialCoordinate(contact_mesh)
x_contact_ref.interpolate(as_vector((X_contact[0], X_contact[1], X_contact[2])))
contact_coords0 = np.asarray(x_contact_ref.dat.data_ro, dtype=float).copy()
nc = contact_coords0.shape[0]
if Q.dim() != nc:
    raise RuntimeError(f"Expected Q.dim()={nc}, got {Q.dim()}.")

I_contact = petsc_to_csr(
    assemble(interpolate(TrialFunction(V), V_contact)).petscmat
).tocsr()
C_gap = I_contact[1::3, :].tocsr()
M_contact = petsc_to_csr(
    assemble(TrialFunction(Q) * TestFunction(Q) * ds_contact).petscmat
).tocsr()
nb = V.dim()
psi_lag = np.zeros(nc)


# material and matrices

# Assemble the 3D Newmark effective matrix A. Both free solve and Schur solve reuse it.

# Small-strain tensor used by the linear elastic material model.
def eps(w):
    return sym(grad(w))


# Linear elastic stress tensor for the ball.
def sigma(w):
    return 2.0 * mu * eps(w) + lmbda * tr(eps(w)) * Identity(3)


# Newmark formula for acceleration at the current trial displacement.
def acceleration_expr(w):
    return (
        w
        - u_old
        - dt * v_old
        - dt**2 * (0.5 - beta_nm) * a_old
    ) / (beta_nm * dt**2)


# Newmark formula for velocity at the current trial displacement.
def velocity_expr(acceleration):
    return v_old + dt * ((1.0 - gamma_nm) * a_old + gamma_nm * acceleration)


# contact geometry

# With a rigid ground, the normal is fixed upward and the gap is distance to the plane.

# Compute the normal contact gap values at all contact degrees of freedom.
def gap_values(uv):
    return contact_coords0[:, 1] + C_gap @ uv - ground_y


# Return the smallest current contact gap.
def min_contact_gap(u_func):
    return float(np.min(gap_values(flat(u_func))))


# Evaluate the ball displacement at the contact points.
def displacement_array(values):
    if hasattr(values, "dat"):
        return flat(values)
    return np.asarray(values, dtype=float).reshape(-1)


# Blend two displacement states and assign the result into a Firedrake Function.
def assign_blended_displacement(out, u_a, u_b, theta):
    set_flat(
        out,
        (1.0 - theta) * displacement_array(u_a) + theta * displacement_array(u_b),
    )


# Backtrack from free flight to find a positive-gap entry state for LVPP.
def positive_lvpp_entry_displacement(ub_free):
    # The free predictor can step too far into contact. This only backs up the contact initial guess.
    current = flat(u_old)
    current_gap_values = gap_values(current)
    free_gap_values = gap_values(ub_free)

    current_gap = float(np.min(current_gap_values))
    free_gap = float(np.min(free_gap_values))

    if free_gap > entry_gap_target:
        return ub_free, "free_positive", 1.0, free_gap

    if current_gap <= entry_gap_target:
        u_entry.assign(u_old)
        return flat(u_entry).copy(), "current_near_contact", 0.0, current_gap

    crossing = free_gap_values <= entry_gap_target
    if not np.any(crossing):
        u_entry.assign(u_old)
        return flat(u_entry).copy(), "current_fallback", 0.0, current_gap

    denom = current_gap_values[crossing] - free_gap_values[crossing]
    valid = denom > 1.0e-30
    if not np.any(valid):
        u_entry.assign(u_old)
        return flat(u_entry).copy(), "current_fallback", 0.0, current_gap

    theta_candidates = (current_gap_values[crossing][valid] - entry_gap_target) / denom[valid]
    theta = float(np.clip(np.min(theta_candidates), 0.0, 1.0))
    theta = float(np.clip(theta * entry_theta_safety, 0.0, 1.0))

    assign_blended_displacement(u_entry, u_old, ub_free, theta)
    entry_gap = min_contact_gap(u_entry)
    return flat(u_entry).copy(), "backtrack", theta, entry_gap


# Initialize psi from the current positive gap using psi = log(gap).
def initialise_psi(uv):
    global psi_lag
    psi_lag = np.log(np.maximum(gap_values(uv), gap_floor))


# Accept a ball displacement and update its Newmark velocity and acceleration.
def accept_displacement(uv):
    set_flat(u_state, uv)
    a_new.interpolate(acceleration_expr(u_state))
    v_new.interpolate(velocity_expr(a_new))
    u_old.assign(u_state)
    v_old.assign(v_new)
    a_old.assign(a_new)


# Convert LVPP psi values into a simple contact-pressure indicator for rendering.
def contact_pressure_indicator(psi, alpha_value, lag):
    dpsi = psi - lag
    pressure = np.maximum(0.0, -dpsi / alpha_value)
    return int(np.count_nonzero(pressure > 1.0e-9)), float(np.max(pressure))


# residuals and contact solver

# Unknowns are [ball displacement, psi]. psi controls slack; psi-lag gives the LVPP pressure.

u_trial = TrialFunction(V)
w_test = TestFunction(V)
A_trial_expr = u_trial / (beta_nm * dt**2)
V_trial_expr = gamma_nm * u_trial / (beta_nm * dt)
A_form = (
    rho * inner(A_trial_expr, w_test)
    + inner(sigma(u_trial), eps(w_test))
    + inner(kv_tau * sigma(V_trial_expr), eps(w_test))
) * dx_main
A_ball = petsc_to_csr(assemble(A_form).petscmat).tocsr()
ball_lu = spla.factorized(A_ball.tocsc())

# Assemble the weak dynamic residual for the ball.
def ball_residual_form():
    a_expr = acceleration_expr(u_state)
    v_expr = velocity_expr(a_expr)
    return (
        rho * inner(a_expr, w_test)
        + inner(sigma(u_state), eps(w_test))
        + inner(kv_tau * sigma(v_expr), eps(w_test))
        - rho * inner(body_force, w_test)
    ) * dx_main


R_zero_form = ball_residual_form()


# Evaluate the ball residual as a flat NumPy vector.
def zero_residual():
    u_state.dat.data[:] = 0.0
    return np.asarray(assemble(R_zero_form).dat.data_ro, dtype=float).reshape(-1)


J_contact = (C_gap.T @ M_contact).tocsr()
C_gap_weak = (M_contact @ C_gap).tocsr()


# Weak residual for the slack equation gap = exp(psi).
def weak_slack_residual(gap, psi):
    return M_contact @ (gap - np.exp(np.clip(psi, -80.0, 40.0)))


# Infinity-norm error of the weak slack equation.
def weak_slack_residual_error(gap, psi):
    return float(np.linalg.norm(weak_slack_residual(gap, psi)) / max(math.sqrt(nc), 1.0))


# Measure the displacement change between two LVPP outer iterates.
def lvpp_increment(ub, ub_prev):
    return math.sqrt(float(np.mean((ub - ub_prev) ** 2)))


# Return the LVPP proximal parameter for the current outer iteration.
def lvpp_alpha(k):
    return min(lvpp_alpha0 * lvpp_alpha_growth**k, lvpp_alpha_max)


# Split the concatenated Newton vector into physical unknown blocks.
def split_unknown(x):
    return x[:nb], x[nb:]


# Assemble the coupled residual for dynamics plus contact.
def contact_residual(ub, psi, rb0, alpha_value, lag):
    dpsi = psi - lag
    gap = gap_values(ub)
    rb = alpha_value * (A_ball @ ub + rb0) + J_contact @ dpsi
    rc = weak_slack_residual(gap, psi)
    return np.concatenate((rb, rc)), gap


# Derivative of the contact gap residual with respect to psi.
def contact_dpsi_matrix(psi):
    return -(
        M_contact @ sp.diags(np.exp(np.clip(psi, -80.0, 40.0)), format="csr")
    ).tocsr()


# Compute one Newton correction by eliminating bulk unknowns with a Schur complement.
def contact_newton_step_schur(residual, psi, alpha_value):
    # Eliminate the ball bulk displacement, then solve the small Schur system on contact dofs.
    rb = residual[:nb]
    rc = residual[nb:]

    base_ball = ball_lu(-rb / alpha_value)
    Ainv_J = solve_columns(ball_lu, J_contact) / alpha_value
    Dpsi = contact_dpsi_matrix(psi).toarray()
    schur = Dpsi - C_gap_weak @ Ainv_J
    rhs_psi = -rc - C_gap_weak @ base_ball
    try:
        dpsi = np.linalg.solve(schur, rhs_psi)
    except np.linalg.LinAlgError:
        dpsi = np.linalg.lstsq(schur, rhs_psi, rcond=None)[0]

    dub = base_ball - Ainv_J @ dpsi
    return np.concatenate((dub, dpsi))


# Run damped Newton iterations for one fixed LVPP outer state.
def saddle_newton_solve(x0, rb0, alpha_value, lag):
    x = x0.copy()
    best_norm = np.inf
    best_x = x.copy()
    best_last = None
    for it in range(newton_max_it):
        ub, psi = split_unknown(x)
        residual, gap = contact_residual(ub, psi, rb0, alpha_value, lag)
        norm = float(np.linalg.norm(residual) / max(math.sqrt(residual.size), 1.0))
        gap_eq = weak_slack_residual_error(gap, psi)
        last = (it + 1, norm, gap_eq, float(np.min(gap)))
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
            best_last = last
        if norm < newton_tol and gap_eq < max(newton_tol, 0.25 * gap_eq_tol):
            return x, last

        dx = contact_newton_step_schur(residual, psi, alpha_value)
        damping = 1.0
        accepted = False
        for _ in range(newton_line_search_steps):
            trial = x + damping * dx
            tub, tpsi = split_unknown(trial)
            trial_residual, trial_gap = contact_residual(tub, tpsi, rb0, alpha_value, lag)
            trial_norm = float(np.linalg.norm(trial_residual) / max(math.sqrt(trial_residual.size), 1.0))
            trial_gap_eq = weak_slack_residual_error(trial_gap, tpsi)
            if trial_norm < norm or trial_norm <= best_norm:
                x = trial
                accepted = True
                break
            if trial_gap_eq < gap_eq and trial_norm < 5.0 * norm:
                x = trial
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            break
    return best_x, best_last


# Run the LVPP contact correction for one time step.
def solve_contact_step(rb0, initial_guess):
    global psi_lag
    initialise_psi(flat(u_old))
    x = np.concatenate((np.asarray(initial_guess, dtype=float).reshape(-1), psi_lag.copy()))

    last = None
    for k in range(lvpp_max_it):
        alpha_value = lvpp_alpha(k)
        lag = psi_lag.copy()
        ub_prev, _ = split_unknown(x)
        x, newton_info = saddle_newton_solve(x, rb0, alpha_value, lag)
        ub, psi = split_unknown(x)
        gap = gap_values(ub)
        gap_eq = weak_slack_residual_error(gap, psi)
        du = lvpp_increment(ub, ub_prev)
        min_gap = float(np.min(gap))
        active, max_pressure = contact_pressure_indicator(psi, alpha_value, lag)
        last = (k, alpha_value, du, gap_eq, min_gap, active, max_pressure, newton_info)

        gap_ok = (not require_gap_equation) or gap_eq < gap_eq_tol
        if du < lvpp_tol and gap_ok and min_gap >= min_gap_tol:
            psi_lag = psi.copy()
            return x, last

        psi_lag = psi.copy()

    k, alpha_value, du, gap_eq, min_gap, active, max_pressure, newton_info = last
    raise RuntimeError(
        "LVPP contact step did not converge: "
        f"it={k + 1}, du={du:.3e}, gap_eq={gap_eq:.3e}, "
        f"min_gap={min_gap:.3e}, LVPP_TOL={lvpp_tol:.3e}, "
        f"GAP_EQ_TOL={gap_eq_tol:.3e}, MIN_GAP_TOL={min_gap_tol:.3e}, "
        f"newton={newton_info}"
    )


# rendering

# The plot is just for the dynamics movie: ball surface triangles plus a fixed ground plane.

frames = []
coords0 = np.asarray(mesh.coordinates.dat.data_ro, dtype=float).copy()
u_plot = Function(mesh.coordinates.function_space(), name="u_plot")
surface_triangles = make_mesh_surface_triangles(mesh)
ground_face = np.asarray(
    [
        [
            (ground_x0, ground_z0, ground_y),
            (ground_x1, ground_z0, ground_y),
            (ground_x1, ground_z1, ground_y),
            (ground_x0, ground_z1, ground_y),
        ]
    ],
    dtype=float,
)

x_min = ground_x0 - 0.12
x_max = ground_x1 + 0.12
z_min = ground_z0 - 0.12
z_max = ground_z1 + 0.12
y_min = ground_y - 0.05
y_max = cy + 1.00 * radius


# Render the current simulation state into the video frame buffer.
def render_frame():
    if not write_video:
        return

    u_plot.interpolate(u_old)
    coords = coords0 + np.asarray(u_plot.dat.data_ro, dtype=float)
    tri_coords = coords[:, [0, 2, 1]][surface_triangles]

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            ground_face,
            facecolor=(0.74, 0.74, 0.72, 0.20),
            edgecolor=(0.08, 0.08, 0.08, 0.18),
            linewidth=0.55,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            tri_coords,
            facecolor=(0.25, 0.52, 0.78, 0.30),
            edgecolor=(0.10, 0.28, 0.46, 0.56),
            linewidth=0.24,
        )
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(y_min, y_max)
    ax.set_box_aspect((x_max - x_min, z_max - z_min, y_max - y_min))
    ax.view_init(elev=10.0, azim=-52.0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)


# time loop

# Each step starts as free flight. The contact corrector starts only when the gap gets close.

v_old.interpolate(initial_velocity)
a_old.interpolate(body_force)
initialise_psi(flat(u_old))

print(
    "3D rigid-ground LVPP Signorini | "
    f"dofs V={V.dim()} Q={Q.dim()} total={V.dim() + Q.dim()} | "
    f"dt={dt_value:.2e} steps={num_steps} | E={young_modulus_value:.2e} | "
    f"kv_tau={kv_tau_value:.1e} | lvpp_tol={lvpp_tol:.1e} min_gap_tol={min_gap_tol:.1e} | "
    f"newton_tol={newton_tol:.1e} | solver=custom Schur Newton | "
    f"activation_gap={activation_gap:.1e} | "
    f"contact_maxh={contact_maxh:.3g} | "
    f"contact_space={contact_space_family}{contact_space_degree} | "
    f"entry_gap_target={entry_gap_target:.1e} | "
    f"entry_backtrack_iters={entry_backtrack_iters} | "
    f"entry_theta_safety={entry_theta_safety:.3f} | "
    f"bottom_label={contact_labels}"
)

simulation_start = time.perf_counter()
contact_steps = 0
max_lvpp_iters = 0
global_min_gap = min_contact_gap(u_old)
accepted_gap_violations = 0
min_trial_gap = math.inf

render_frame()

for step in range(1, num_steps + 1):
    rb0 = zero_residual()
    ub_free = ball_lu(-rb0)
    trial_gap = float(np.min(gap_values(ub_free)))
    min_trial_gap = min(min_trial_gap, trial_gap)
    lvpp_info = None

    if trial_gap > activation_gap:
        accept_displacement(ub_free)
        min_gap = min_contact_gap(u_old)
    else:
        contact_steps += 1
        initial_guess, _, _, _ = positive_lvpp_entry_displacement(ub_free)
        x, lvpp_info = solve_contact_step(rb0, initial_guess)
        ub, _ = split_unknown(x)
        accept_displacement(ub)
        min_gap = min_contact_gap(u_old)
        max_lvpp_iters = max(max_lvpp_iters, lvpp_info[0] + 1)

    if min_gap <= min_gap_tol:
        accepted_gap_violations += 1
    global_min_gap = min(global_min_gap, min_gap)

    if step % save_every == 0 or step == num_steps:
        render_frame()



# output

if write_video and frames:
    imageio.mimsave(
        video_name,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    print(f"MP4 written: {os.path.abspath(video_name)}")

print(
    "simulation complete | "
    f"contact_steps={contact_steps} | max_lvpp_iters={max_lvpp_iters} | "
    f"global_min_gap={global_min_gap:.3e} | "
    f"min_trial_gap={min_trial_gap:.3e} | "
    f"accepted_gap_violations={accepted_gap_violations} | "
    f"min_gap_tol={min_gap_tol:.3e} | "
    f"wall={time.perf_counter() - simulation_start:.2f}s"
)
