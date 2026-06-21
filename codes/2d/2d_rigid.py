from firedrake import *
from netgen.occ import *

import math
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def env_float(name, default): return float(os.environ.get(name, default))


def env_int(name, default): return int(os.environ.get(name, default))


def env_bool(name, default):
    raw = os.environ.get(name)
    return bool(default) if raw is None else raw.lower() not in {"0", "false", "no", "off"}


# Parameters
radius = env_float("BALL_RADIUS", 0.20)
cx = env_float("BALL_CENTER_X", 0.0)
ground_y_value = env_float("GROUND_Y", 0.0)
initial_gap = env_float("INITIAL_GAP", 0.45)
cy = ground_y_value + radius + initial_gap

mesh_degree = env_int("MESH_DEGREE", 2)
fe_degree = env_int("FE_DEGREE", 1)
maxh = env_float("MESH_MAXH", 0.025)
quad = {"quadrature_degree": env_int("QUADRATURE_DEGREE", 3)}

rho = Constant(env_float("DENSITY", 1.0))
young_modulus_value = env_float("YOUNG_MODULUS", 1.0e4)
E = Constant(young_modulus_value)
nu = Constant(env_float("POISSON_RATIO", 0.30))
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
body_force = Constant((0.0, env_float("GRAVITY", -9.81)))
kv_tau = Constant(env_float("KV_TAU", 5.0e-4))

dt_value = env_float("SIM_DT", 2.0e-4)
num_steps = env_int("SIM_STEPS", 1600)
save_every = env_int("SAVE_EVERY", 8)
dt = Constant(dt_value)

gamma_value = env_float("NEWMARK_GAMMA", 0.5)
beta_value = env_float("NEWMARK_BETA", 0.25 * (gamma_value + 0.5) ** 2)
beta_nm = Constant(beta_value)
gamma_nm = Constant(gamma_value)
initial_velocity = Constant((0.0, env_float("INITIAL_VY", -0.65)))

lvpp_alpha0 = env_float("LVPP_ALPHA0", 2.0e-5)
lvpp_alpha_growth = env_float("LVPP_ALPHA_GROWTH", 1.35)
lvpp_alpha_max = env_float("LVPP_ALPHA_MAX", 1.0e-4)
alpha = Constant(lvpp_alpha0)
lvpp_tol = env_float("LVPP_TOL", 1.0e-5)
if lvpp_tol <= 0.0:
    raise ValueError("LVPP_TOL must be positive.")
gap_eq_tol = env_float("GAP_EQ_TOL", lvpp_tol)
penetration_tol = env_float("PENETRATION_TOL", 1.0e-12)
if penetration_tol < 0.0:
    raise ValueError("PENETRATION_TOL must be nonnegative.")
min_gap_tol = env_float("MIN_GAP_TOL", -penetration_tol)
gap_floor = env_float("LVPP_GAP_FLOOR", 1.0e-10)
lvpp_max_it = env_int("LVPP_MAX_IT", 100)
activation_gap = env_float("LVPP_ACTIVATION_GAP", 2.0e-3)
require_gap_equation = env_bool("REQUIRE_GAP_EQUATION", False)

video_name = os.environ.get("SIM_VIDEO_NAME", "2d_rigid.mp4")
write_video = env_bool("WRITE_VIDEO", True)
fps = env_int("VIDEO_FPS", 24)
ground_x0 = env_float("GROUND_X0", cx - 1.65 * radius)
ground_x1 = env_float("GROUND_X1", cx + 1.65 * radius)


# Mesh
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
mesh = Mesh(ngmesh, distribution_parameters=no_overlap)
mesh = Mesh(mesh.curve_field(mesh_degree), distribution_parameters=no_overlap)

contact_labels = [i + 1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bottom"]
if not contact_labels:
    raise RuntimeError("Could not find the Netgen boundary label 'bottom'.")

contact_mesh = Submesh(mesh, 1, contact_labels)

dx_main = Measure("dx", mesh, metadata=quad)
ds_contact = Measure("dx", contact_mesh, metadata=quad, intersect_measures=(Measure("ds", mesh),))


# Spaces and state
V = VectorFunctionSpace(mesh, "CG", fe_degree)
Q = FunctionSpace(contact_mesh, os.environ.get("CONTACT_SPACE_FAMILY", "CG"), env_int("CONTACT_SPACE_DEGREE", fe_degree))
W = V * Q

z = Function(W, name="lvpp_state")
u, psi = split(z)
z_test = TestFunction(W)
v_test, q_test = split(z_test)
u_h, psi_h = z.subfunctions

u_old, v_old, a_old, u_free, u_prev, v_new, a_new = [
    Function(V, name=name)
    for name in ("u_old", "v_old", "a_old", "u_free", "u_prev", "v_new", "a_new")
]
psi_lag = Function(Q, name="psi_lag")

V_contact = VectorFunctionSpace(contact_mesh, "CG", fe_degree)
u_contact = Function(V_contact, name="u_contact")
x_contact_ref = Function(V_contact, name="x_contact_ref")
X_contact = SpatialCoordinate(contact_mesh)
x_contact_ref.interpolate(as_vector((X_contact[0], X_contact[1])))
contact_coords0 = np.asarray(x_contact_ref.dat.data_ro, dtype=float).copy()

X = SpatialCoordinate(mesh)


def eps(w):
    return sym(grad(w))


def sigma(w):
    return 2.0 * mu * eps(w) + lmbda * tr(eps(w)) * Identity(2)


def acceleration_expr(w):
    return (w - u_old - dt * v_old - dt**2 * (0.5 - beta_nm) * a_old) / (beta_nm * dt**2)


def velocity_expr(acceleration):
    return v_old + dt * ((1.0 - gamma_nm) * a_old + gamma_nm * acceleration)


def contact_gap_values(u_func):
    u_contact.interpolate(u_func)
    values = np.asarray(u_contact.dat.data_ro, dtype=float)
    return contact_coords0[:, 1] + values[:, 1] - ground_y_value


def min_contact_gap(u_func): return float(np.min(contact_gap_values(u_func)))


def initialise_psi(u_func):
    u_contact.interpolate(u_func)
    physical_gap_expr = X_contact[1] + u_contact[1] - Constant(ground_y_value)
    psi_lag.interpolate(ln(max_value(physical_gap_expr, Constant(gap_floor))))


def accept_displacement(u_func):
    a_new.interpolate(acceleration_expr(u_func))
    v_new.interpolate(velocity_expr(a_new))
    u_old.assign(u_func)
    v_old.assign(v_new)
    a_old.assign(a_new)


def contact_pressure_indicator(alpha_value):
    dpsi = np.asarray(psi_h.dat.data_ro - psi_lag.dat.data_ro, dtype=float)
    pressure = np.maximum(0.0, -dpsi / alpha_value)
    return int(np.count_nonzero(pressure > 1.0e-9)), float(np.max(pressure))


# Free flight and contact subproblems
u_trial = TrialFunction(V)
w_test = TestFunction(V)
a_free = acceleration_expr(u_trial)
v_free = velocity_expr(a_free)
F_free = (
    rho * inner(a_free, w_test)
    + inner(sigma(u_trial), eps(w_test))
    + inner(kv_tau * sigma(v_free), eps(w_test))
    - rho * inner(body_force, w_test)
) * dx_main

lu_params = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": os.environ.get("LU_SOLVER", "mumps"),
}

free_solver = LinearVariationalSolver(
    LinearVariationalProblem(lhs(F_free), rhs(F_free), u_free),
    solver_parameters=lu_params,
)

a_expr = acceleration_expr(u)
v_expr = velocity_expr(a_expr)
R_mech = alpha * (
    rho * inner(a_expr, v_test)
    + inner(sigma(u), eps(v_test))
    + inner(kv_tau * sigma(v_expr), eps(v_test))
    - rho * inner(body_force, v_test)
) * dx_main

# LVPP Signorini contact
phi1 = X[1] - Constant(ground_y_value)
physical_gap = phi1 - dot(u, Constant((0.0, -1.0)))

contact_potential = (psi - psi_lag) * physical_gap * ds_contact
R_contact = derivative(contact_potential, z, z_test) - exp(psi) * q_test * ds_contact

contact_solver = NonlinearVariationalSolver(
    NonlinearVariationalProblem(R_mech + R_contact, z),
    solver_parameters={
        **lu_params,
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_linesearch_damping": env_float("NEWTON_DAMPING", 0.65),
        "snes_atol": env_float("NEWTON_ATOL", 1.0e-10),
        "snes_rtol": env_float("NEWTON_RTOL", 1.0e-8),
        "snes_max_it": env_int("NEWTON_MAX_IT", 60),
        "snes_divergence_tolerance": 1.0e12,
    },
)


def weak_slack_residual_error():
    return math.sqrt(max(float(assemble((physical_gap - exp(psi_h)) ** 2 * ds_contact)), 0.0))


def lvpp_increment():
    return math.sqrt(max(float(assemble(inner(u_h - u_prev, u_h - u_prev) * dx_main)), 0.0))


def lvpp_alpha(k): return min(lvpp_alpha0 * lvpp_alpha_growth**k, lvpp_alpha_max)


def solve_contact_step():
    initialise_psi(u_old)
    u_h.assign(u_free)
    psi_h.assign(psi_lag)

    last = None
    for k in range(lvpp_max_it):
        alpha_value = lvpp_alpha(k)
        alpha.assign(alpha_value)
        u_prev.assign(u_h)

        contact_solver.solve()

        du = lvpp_increment()
        gap_eq = weak_slack_residual_error()
        min_gap = min_contact_gap(u_h)
        active, max_pressure = contact_pressure_indicator(alpha_value)
        last = (k, alpha_value, du, gap_eq, min_gap, active, max_pressure)

        gap_ok = (not require_gap_equation) or gap_eq < gap_eq_tol
        if du < lvpp_tol and gap_ok and min_gap > min_gap_tol:
            return last

        psi_lag.assign(psi_h)

    k, alpha_value, du, gap_eq, min_gap, active, max_pressure = last
    raise RuntimeError(
        "LVPP contact step did not converge: "
        f"it={k + 1}, du={du:.3e}, gap_eq={gap_eq:.3e}, "
        f"min_gap={min_gap:.3e}, LVPP_TOL={lvpp_tol:.3e}, "
        f"GAP_EQ_TOL={gap_eq_tol:.3e}, MIN_GAP_TOL={min_gap_tol:.3e}"
    )


# Rendering
frames = []
P1_render = FunctionSpace(mesh, "CG", 1)
V_render = VectorFunctionSpace(mesh, "CG", 1)
x_render = Function(V_render, name="x_render")
u_render = Function(V_render, name="u_render")
x_render.interpolate(as_vector((X[0], X[1])))
mesh_coords0 = np.asarray(x_render.dat.data_ro, dtype=float).copy()
mesh_triangles = np.asarray(P1_render.cell_node_map().values, dtype=int)
ground_line_x = np.asarray([ground_x0, ground_x1], dtype=float)
ground_line_y = np.full(2, ground_y_value, dtype=float)


def ordered_boundary_loops(triangles):
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

    unused = set(edges)
    loops = []
    while unused:
        start, current = unused.pop()
        loop = [start, current]
        previous = start
        while current != start:
            next_node = next(
                (
                    node
                    for node in neighbors[current]
                    if node != previous and tuple(sorted((current, node))) in unused
                ),
                None,
            )
            if next_node is None:
                close = tuple(sorted((current, start)))
                if close in unused:
                    unused.remove(close)
                    loop.append(start)
                break
            unused.remove(tuple(sorted((current, next_node))))
            loop.append(next_node)
            previous, current = current, next_node
        if loop[-1] != start:
            loop.append(start)
        loops.append(np.asarray(loop, dtype=int))
    return loops


mesh_boundary_loops = ordered_boundary_loops(mesh_triangles)


def render_frame():
    if not write_video:
        return

    u_render.interpolate(u_old)
    coords = mesh_coords0 + np.asarray(u_render.dat.data_ro, dtype=float)
    tri_coords = coords[mesh_triangles]

    fig, ax = plt.subplots(figsize=(5, 7), dpi=100)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.add_collection(
        PolyCollection(
            tri_coords,
            facecolor=(0.80, 0.90, 0.96, 0.36),
            edgecolor=(0.50, 0.70, 0.84, 0.18),
            linewidth=0.26,
            antialiased=True,
        )
    )
    for loop in mesh_boundary_loops:
        loop_coords = coords[loop]
        ax.plot(
            loop_coords[:, 0],
            loop_coords[:, 1],
            color=(0.19, 0.49, 0.70, 0.95),
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.plot(ground_line_x, ground_line_y, color="black", linewidth=2.2, solid_capstyle="butt")

    ax.set_aspect("equal", adjustable="box")
    x_margin = 0.45 * radius
    ax.set_xlim(ground_x0 - x_margin, ground_x1 + x_margin)
    ax.set_ylim(ground_y_value - 0.16 * radius, cy + 1.05 * radius)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)


v_old.interpolate(initial_velocity)
a_old.interpolate(body_force)
initialise_psi(u_old)
psi_h.assign(psi_lag)

print(
    "2D rigid-ground LVPP Signorini | "
    f"dofs V={V.dim()} Q={Q.dim()} W={W.dim()} | "
    f"dt={dt_value:.2e} steps={num_steps} | E={young_modulus_value:.2e} | "
    f"lvpp_tol={lvpp_tol:.1e} min_gap_tol={min_gap_tol:.1e} | "
    f"activation_gap={activation_gap:.1e} | "
    "gap_form=Signorini weak slack | "
    "mesh=OCC glued half-disk | bottom_label=bottom | "
    f"contact_labels={contact_labels}"
)

simulation_start = time.perf_counter()
contact_steps = 0
max_lvpp_iters = 0
global_min_gap = min_contact_gap(u_old)
log_every = env_int("LOG_EVERY", 50)
log_contact_every = env_int("LOG_CONTACT_EVERY", 10)

render_frame()

for step in range(1, num_steps + 1):
    free_solver.solve()
    trial_gap = min_contact_gap(u_free)

    if trial_gap > activation_gap:
        accept_displacement(u_free)
        lvpp_info = None
        min_gap = trial_gap
    else:
        contact_steps += 1
        lvpp_info = solve_contact_step()
        accept_displacement(u_h)
        min_gap = min_contact_gap(u_old)
        max_lvpp_iters = max(max_lvpp_iters, lvpp_info[0] + 1)

    global_min_gap = min(global_min_gap, min_gap)

    if step % save_every == 0 or step == num_steps:
        render_frame()

    should_log = step % log_every == 0
    should_log = should_log or (
        lvpp_info is not None
        and (contact_steps == 1 or step % log_contact_every == 0)
    )
    if should_log:
        if lvpp_info is None:
            print(f"step {step:04d} | free    | trial_gap={trial_gap:.3e}")
        else:
            k, alpha_value, du, gap_eq, _, active, max_pressure = lvpp_info
            print(
                f"step {step:04d} | contact | min_gap={min_gap:.3e} | "
                f"lvpp={k + 1} | alpha={alpha_value:.2e} | du={du:.3e} | "
                f"gap_eq={gap_eq:.3e} | active={active}/{Q.dim()} | "
                f"pmax={max_pressure:.3e}"
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

print(
    "simulation complete | "
    f"contact_steps={contact_steps} | max_lvpp_iters={max_lvpp_iters} | "
    f"global_min_gap={global_min_gap:.3e} | wall={time.perf_counter() - simulation_start:.2f}s"
)
