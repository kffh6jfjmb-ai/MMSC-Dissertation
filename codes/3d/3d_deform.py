"""
3D deformable ground case.

Here both the ball and the ground move, and both are part of the contact solve.
Each step starts with a no-contact prediction. Once contact activates, ball displacement,
ground displacement, and psi are solved together in one LVPP/Newton system.
The normal is the Piola normal from the deformed ground.
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
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

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

# Geometry, material, time step, mesh, and contact tolerances live here.

radius = 0.20
cx = 0.0
cz = 0.0
ground_y = 0.0
initial_gap = 0.10
cy = ground_y + radius + initial_gap

mesh_degree = 2
fe_degree = 2
maxh = 0.08
contact_maxh = 0.035
contact_threshold = -0.75
contact_space_family = "CG"
contact_space_degree = 1
quad = {"quadrature_degree": 3}

rho_ball = Constant(1.0)
E_ball_value = 3.0e2
E_ball = Constant(E_ball_value)
nu_ball = Constant(0.30)
mu_ball = E_ball / (2.0 * (1.0 + nu_ball))
lmbda_ball = E_ball * nu_ball / ((1.0 + nu_ball) * (1.0 - 2.0 * nu_ball))
ball_body_force = Constant((0.0, -9.81, 0.0))
ball_kv_tau = Constant(8.0e-4)

rho_ground = Constant(1.0)
E_ground_value = 1.5e3
E_ground = Constant(E_ground_value)
nu_ground = Constant(0.30)
mu_ground = E_ground / (2.0 * (1.0 + nu_ground))
lmbda_ground = E_ground * nu_ground / ((1.0 + nu_ground) * (1.0 - 2.0 * nu_ground))
ground_body_force = Constant((0.0, 0.0, 0.0))
ground_kv_tau = Constant(2.0e-3)

dt_value = 2.0e-4
num_steps = 1500
save_every = 30
dt = Constant(dt_value)

gamma_value = 0.5
beta_value = 0.25 * (gamma_value + 0.5) ** 2
gamma_nm = Constant(gamma_value)
beta_nm = Constant(beta_value)
initial_velocity = Constant((0.0, -1.20, 0.0))

lvpp_alpha0 = 2.0e-3
lvpp_alpha_growth = 1.15
lvpp_alpha_max = 2.0e-2
lvpp_tol = 1.0e-6
gap_eq_tol = 1.0e-8
penetration_tol = 1.0e-16
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

newton_tol = 1.0e-10
newton_max_it = 35
newton_line_search_steps = 12
contact_linear_solver = "schur"
if contact_linear_solver not in {"schur", "direct"}:
    raise ValueError("CONTACT_LINEAR_SOLVER must be schur or direct.")

video_name = "3d_deformable_ground.mp4"
write_video = True
fps = 24

ground_x0 = cx - 1.65 * radius
ground_x1 = cx + 1.65 * radius
ground_z0 = cz - 1.65 * radius
ground_z1 = cz + 1.65 * radius
ground_depth = 0.20
ground_nx = 12
ground_ny = 3
ground_nz = 12
surface_eps = 1.0e-10


# helpers and material

# Helpers, material law, and VertexOnlyMesh interpolation matrices. Contact values come through these.

# Small-strain tensor used by the linear elastic material model.
def eps(w):
    return sym(grad(w))


# Linear elastic stress tensor for the ball.
def sigma(w, mu, lmbda):
    return 2.0 * mu * eps(w) + lmbda * tr(eps(w)) * Identity(3)


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


# Build a VertexOnlyMesh interpolation matrix and reorder rows to match our point order.
def ordered_vom_interpolation_matrix(mesh_obj, source_space, points, tol=1.0e-8):
    points = np.asarray(points, dtype=float)
    vom = VertexOnlyMesh(mesh_obj, points, missing_points_behaviour="error")
    target_space = VectorFunctionSpace(vom, "DG", 0, dim=3)
    raw = petsc_to_csr(
        assemble(interpolate(TrialFunction(source_space), target_space)).petscmat
    ).tocsr()
    vom_coords = np.asarray(vom.coordinates.dat.data_ro, dtype=float)
    distances, vom_ids = cKDTree(vom_coords).query(points, k=1)
    if len(distances) and float(np.max(distances)) > tol:
        raise RuntimeError("VertexOnlyMesh point ordering failed.")
    rows = np.empty(3 * len(vom_ids), dtype=int)
    rows[0::3] = 3 * vom_ids
    rows[1::3] = 3 * vom_ids + 1
    rows[2::3] = 3 * vom_ids + 2
    return raw[rows, :].tocsr()


# Build gradient interpolation matrices on a VertexOnlyMesh in our point order.
def ordered_vom_gradient_matrix(mesh_obj, source_space, points, tol=1.0e-8):
    points = np.asarray(points, dtype=float)
    vom = VertexOnlyMesh(mesh_obj, points, missing_points_behaviour="error")
    target_space = TensorFunctionSpace(vom, "DG", 0, shape=(3, 3))
    raw = petsc_to_csr(
        assemble(interpolate(grad(TrialFunction(source_space)), target_space)).petscmat
    ).tocsr()
    vom_coords = np.asarray(vom.coordinates.dat.data_ro, dtype=float)
    distances, vom_ids = cKDTree(vom_coords).query(points, k=1)
    if len(distances) and float(np.max(distances)) > tol:
        raise RuntimeError("VertexOnlyMesh point ordering failed.")
    rows = np.empty(9 * len(vom_ids), dtype=int)
    for i in range(9):
        rows[i::9] = 9 * vom_ids + i
    return raw[rows, :].tocsr()


# Normalize row vectors while protecting against nearly zero-length rows.
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


# List each triangle edge once so the contact mesh can be drawn cleanly.
def unique_triangle_edges(triangles):
    edges = set()
    for tri in np.asarray(triangles, dtype=int):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    return np.asarray(sorted(edges), dtype=int)


# mesh and spaces

# The ball is a 3D volume mesh, the ground is a regular box, and the contact mesh is the lower cap.

no_overlap = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}
ball_mesh, contact_mesh, contact_labels = build_bottom_cap_sphere(
    radius,
    (cx, cy, cz),
    contact_threshold,
    maxh,
    mesh_degree,
    contact_maxh,
)

ground_lx = ground_x1 - ground_x0
ground_lz = ground_z1 - ground_z0
ground_mesh = BoxMesh(
    ground_nx,
    ground_ny,
    ground_nz,
    ground_lx,
    ground_depth,
    ground_lz,
    distribution_parameters=no_overlap,
)
ground_mesh.coordinates.dat.data[:, 0] += ground_x0
ground_mesh.coordinates.dat.data[:, 1] += ground_y - ground_depth
ground_mesh.coordinates.dat.data[:, 2] += ground_z0
ground_bottom_id = 3

dx_ball = Measure(
    "dx",
    ball_mesh,
    metadata=quad,
    intersect_measures=(Measure("dx", ball_mesh), Measure("ds", contact_mesh)),
)
dx_ground = Measure("dx", ground_mesh, metadata=quad)
dx_contact = Measure(
    "dx",
    contact_mesh,
    metadata=quad,
    intersect_measures=(Measure("ds", ball_mesh),),
)

V = VectorFunctionSpace(ball_mesh, "CG", fe_degree)
Vc = VectorFunctionSpace(contact_mesh, contact_space_family, contact_space_degree)
Q = FunctionSpace(contact_mesh, contact_space_family, contact_space_degree)
G = VectorFunctionSpace(ground_mesh, "CG", 1)

u_old = Function(V, name="u_old")
v_old = Function(V, name="v_old")
a_old = Function(V, name="a_old")
u_state = Function(V, name="u_state")
ball_a_new = Function(V, name="ball_a_new")
ball_v_new = Function(V, name="ball_v_new")

ground_u_old = Function(G, name="ground_u_old")
ground_v_old = Function(G, name="ground_v_old")
ground_a_old = Function(G, name="ground_a_old")
ground_state = Function(G, name="ground_state")
ground_a_new = Function(G, name="ground_a_new")
ground_v_new = Function(G, name="ground_v_new")

v_old.interpolate(initial_velocity)
a_old.interpolate(ball_body_force)
ground_a_old.interpolate(ground_body_force)
ground_bc = DirichletBC(G, Constant((0.0, 0.0, 0.0)), ground_bottom_id)


# matrices and free solve

# Assemble the no-contact Newmark systems first. The free predictor uses these matrices.

# Newmark acceleration formula for the ball displacement.
def ball_accel_expr(w):
    return (w - u_old - dt * v_old - dt**2 * (0.5 - beta_nm) * a_old) / (beta_nm * dt**2)


# Newmark velocity formula for the ball displacement.
def ball_vel_expr(acc):
    return v_old + dt * ((1.0 - gamma_nm) * a_old + gamma_nm * acc)


# Newmark acceleration formula for the ground displacement.
def ground_accel_expr(w):
    return (w - ground_u_old - dt * ground_v_old - dt**2 * (0.5 - beta_nm) * ground_a_old) / (beta_nm * dt**2)


# Newmark velocity formula for the ground displacement.
def ground_vel_expr(acc):
    return ground_v_old + dt * ((1.0 - gamma_nm) * ground_a_old + gamma_nm * acc)


du = TrialFunction(V)
vb = TestFunction(V)
ball_a_trial = du / (beta_nm * dt**2)
ball_v_trial = gamma_nm * du / (beta_nm * dt)
A_ball_form = (
    rho_ball * inner(ball_a_trial, vb)
    + inner(sigma(du, mu_ball, lmbda_ball), eps(vb))
    + inner(ball_kv_tau * sigma(ball_v_trial, mu_ball, lmbda_ball), eps(vb))
) * dx_ball
A_ball = petsc_to_csr(assemble(A_ball_form).petscmat).tocsr()
ball_lu = spla.factorized(A_ball.tocsc())

dg = TrialFunction(G)
wg = TestFunction(G)
ground_a_trial = dg / (beta_nm * dt**2)
ground_v_trial = gamma_nm * dg / (beta_nm * dt)
A_ground_form = (
    rho_ground * inner(ground_a_trial, wg)
    + inner(sigma(dg, mu_ground, lmbda_ground), eps(wg))
    + inner(ground_kv_tau * sigma(ground_v_trial, mu_ground, lmbda_ground), eps(wg))
) * dx_ground
A_ground = petsc_to_csr(assemble(A_ground_form, bcs=ground_bc).petscmat).tocsr()
ground_lu = spla.factorized(A_ground.tocsc())


# Assemble the weak dynamic residual for the ball.
def ball_residual_form():
    a_expr = ball_accel_expr(u_state)
    v_expr = ball_vel_expr(a_expr)
    return (
        rho_ball * inner(a_expr, vb)
        + inner(sigma(u_state, mu_ball, lmbda_ball), eps(vb))
        + inner(ball_kv_tau * sigma(v_expr, mu_ball, lmbda_ball), eps(vb))
        - rho_ball * inner(ball_body_force, vb)
    ) * dx_ball


# Assemble the weak dynamic residual for the deformable ground.
def ground_residual_form():
    a_expr = ground_accel_expr(ground_state)
    v_expr = ground_vel_expr(a_expr)
    return (
        rho_ground * inner(a_expr, wg)
        + inner(sigma(ground_state, mu_ground, lmbda_ground), eps(wg))
        + inner(ground_kv_tau * sigma(v_expr, mu_ground, lmbda_ground), eps(wg))
        - rho_ground * inner(ground_body_force, wg)
    ) * dx_ground


R_ball_zero_form = ball_residual_form()
R_ground_zero_form = ground_residual_form()


# Evaluate the ball residual as a flat NumPy vector.
def zero_ball_residual():
    u_state.dat.data[:] = 0.0
    return np.asarray(assemble(R_ball_zero_form).dat.data_ro, dtype=float).reshape(-1)


# Evaluate the ground residual as a flat NumPy vector.
def zero_ground_residual():
    ground_state.dat.data[:] = 0.0
    return np.asarray(assemble(R_ground_zero_form, bcs=ground_bc).dat.data_ro, dtype=float).reshape(-1)


# contact geometry

# This is where deformed ground normal, relative vector, and gap are computed.
# The normal comes from F^{-T}N, so it follows the ground deformation.

Xc = SpatialCoordinate(contact_mesh)
contact_ref = Function(Vc, name="contact_ref")
contact_ref.interpolate(as_vector((Xc[0], Xc[1], Xc[2])))
contact_coords = np.asarray(contact_ref.dat.data_ro, dtype=float).copy()
nc = contact_coords.shape[0]
if Q.dim() != nc:
    raise RuntimeError(f"Expected Q.dim()={nc}, got {Q.dim()}.")

I_ball_contact = petsc_to_csr(
    assemble(interpolate(TrialFunction(V), Vc)).petscmat
).tocsr()
B_ball_components = tuple(I_ball_contact[i::3, :].tocsr() for i in range(3))


# Return the ground surface sample points paired with the contact points.
def ground_trace_points(dx_offset=0.0, dz_offset=0.0):
    x = np.clip(
        contact_coords[:, 0] + dx_offset,
        ground_x0 + surface_eps,
        ground_x1 - surface_eps,
    )
    z = np.clip(
        contact_coords[:, 2] + dz_offset,
        ground_z0 + surface_eps,
        ground_z1 - surface_eps,
    )
    return np.column_stack((x, np.full(nc, ground_y - surface_eps), z))


ground_points = ground_trace_points()
I_ground_contact = ordered_vom_interpolation_matrix(ground_mesh, G, ground_points)
B_ground_components = tuple(I_ground_contact[i::3, :].tocsr() for i in range(3))
I_ground_gradient = ordered_vom_gradient_matrix(ground_mesh, G, ground_points)
B_ground_gradient = tuple(
    tuple(I_ground_gradient[(3 * i + j)::9, :].tocsr() for j in range(3))
    for i in range(3)
)
M_contact = petsc_to_csr(
    assemble(TrialFunction(Q) * TestFunction(Q) * dx_contact).petscmat
).tocsr()

nb = V.dim()
ng = G.dim()
psi_lag = np.zeros(nc)
reference_contact_normal = np.asarray((0.0, -1.0, 0.0))


# Evaluate the ball displacement on the contact mesh.
def contact_ball_displacement(ub):
    return (I_ball_contact @ ub).reshape((-1, 3))


# Evaluate the ground displacement at the paired ground contact points.
def contact_ground_displacement(ug):
    return (I_ground_contact @ ug).reshape((-1, 3))


# Compute deformed ground normals and geometry used by the gap equation.
def deformed_ground_geometry(ug):
    grad_u = (I_ground_gradient @ ug).reshape((-1, 3, 3))
    F = grad_u + np.eye(3)[None, :, :]
    F_inv_t = np.linalg.inv(F).transpose(0, 2, 1)
    normal_raw = -np.einsum("nij,j->ni", F_inv_t, reference_contact_normal)
    normal, normal_norm = normalized_rows(normal_raw, np.asarray((0.0, 1.0, 0.0)))
    orientation = np.ones(nc)
    flip = normal[:, 1] < 0.0
    orientation[flip] = -1.0
    normal = normal * orientation[:, None]
    normal_raw = normal_raw * orientation[:, None]
    return normal, normal_raw, F_inv_t, normal_norm


# Project a vector interpolation matrix onto the supplied normal directions.
def normal_projected_matrix(component_matrices, normals):
    projected = sp.diags(normals[:, 0], format="csr") @ component_matrices[0]
    projected += sp.diags(normals[:, 1], format="csr") @ component_matrices[1]
    projected += sp.diags(normals[:, 2], format="csr") @ component_matrices[2]
    return projected.tocsr()


# Combine gradient interpolation blocks using per-point tensor weights.
def weighted_gradient_matrix(weights):
    projected = sp.csr_matrix((nc, ng))
    for i in range(3):
        for j in range(3):
            projected += sp.diags(weights[:, i, j], format="csr") @ B_ground_gradient[i][j]
    return projected.tocsr()


# Build all geometry needed by the contact gap calculation.
def contact_geometry(ub, ug):
    normal, normal_raw, F_inv_t, normal_norm = deformed_ground_geometry(ug)
    relative = (
        contact_coords
        + contact_ball_displacement(ub)
        - ground_points
        - contact_ground_displacement(ug)
    )
    gap = np.einsum("ij,ij->i", relative, normal)
    return gap, normal, relative, normal_raw, F_inv_t, normal_norm


# Build linear maps from displacement increments to normal contact quantities.
def contact_projection_matrices(normal, relative, normal_raw, F_inv_t, normal_norm):
    C_ball = normal_projected_matrix(B_ball_components, normal)
    B_ground_n = normal_projected_matrix(B_ground_components, normal)

    safe_s = np.maximum(normal_norm, 1.0e-14)
    projected_relative = relative - np.einsum("ij,ij->i", relative, normal)[:, None] * normal
    w = projected_relative / safe_s[:, None]
    a = np.einsum("nji,nj->ni", F_inv_t, w)
    grad_weights = -normal_raw[:, :, None] * a[:, None, :]
    normal_term = weighted_gradient_matrix(grad_weights)

    C_ground = (-B_ground_n + normal_term).tocsr()
    J_ball = (C_ball.T @ M_contact).tocsr()
    J_ground = (C_ground.T @ M_contact).tocsr()
    C_ball_weak = (M_contact @ C_ball).tocsr()
    C_ground_weak = (M_contact @ C_ground).tocsr()
    return J_ball, J_ground, C_ball_weak, C_ground_weak


# Compute the normal contact gap values at all contact degrees of freedom.
def gap_values(ub, ug):
    return contact_geometry(ub, ug)[0]


# contact residuals and linear systems

# The residual has ball, ground, and gap-equation blocks. psi is still the log-slack variable.

# Weak residual for the slack equation gap = exp(psi).
def weak_slack_residual(gap, psi):
    return M_contact @ (gap - np.exp(np.clip(psi, -80.0, 40.0)))


# Infinity-norm error of the weak slack equation.
def weak_slack_error(gap, psi):
    residual = weak_slack_residual(gap, psi)
    return float(np.linalg.norm(residual) / max(math.sqrt(nc), 1.0))


# Initialize the lagged LVPP psi from the current positive gap.
def initialise_lvpp_lag(ub, ug):
    global psi_lag
    psi_lag = np.log(np.maximum(gap_values(ub, ug), gap_floor))


# Split the concatenated Newton vector into physical unknown blocks.
def split_unknown(x):
    return x[:nb], x[nb : nb + ng], x[nb + ng :]


# Assemble the coupled residual for dynamics plus contact.
def contact_residual(ub, ug, psi, rb0, rg0, alpha_value, lag):
    dpsi = psi - lag
    gap, normal, relative, normal_raw, F_inv_t, normal_norm = contact_geometry(ub, ug)
    J_ball, J_ground, _, _ = contact_projection_matrices(
        normal,
        relative,
        normal_raw,
        F_inv_t,
        normal_norm,
    )
    rb = alpha_value * (A_ball @ ub + rb0) + J_ball @ dpsi
    rg = alpha_value * (A_ground @ ug + rg0) + J_ground @ dpsi
    rc = weak_slack_residual(gap, psi)
    geometry = (normal, relative, normal_raw, F_inv_t, normal_norm)
    return np.concatenate((rb, rg, rc)), gap, geometry


# Derivative of the contact gap residual with respect to psi.
def contact_dpsi_matrix(psi):
    return -(
        M_contact @ sp.diags(np.exp(np.clip(psi, -80.0, 40.0)), format="csr")
    ).tocsr()


# Assemble the full sparse saddle system for a direct Newton step.
def contact_sparse_linear_system(psi, alpha_value, geometry):
    J_ball, J_ground, C_ball_weak, C_ground_weak = contact_projection_matrices(*geometry)
    return sp.bmat(
        [
            [alpha_value * A_ball, None, J_ball],
            [None, alpha_value * A_ground, J_ground],
            [C_ball_weak, C_ground_weak, contact_dpsi_matrix(psi)],
        ],
        format="csr",
    )


# Compute one Newton correction by eliminating bulk unknowns with a Schur complement.
def contact_newton_step_schur(residual, psi, alpha_value, geometry):
    # Schur first eliminates ball/ground bulk displacement, then solves on contact dofs.
    J_ball, J_ground, C_ball_weak, C_ground_weak = contact_projection_matrices(*geometry)
    rb = residual[:nb]
    rg = residual[nb : nb + ng]
    rc = residual[nb + ng :]

    base_ball = ball_lu(-rb / alpha_value)
    base_ground = ground_lu(-rg / alpha_value)
    Ainv_J_ball = solve_columns(ball_lu, J_ball) / alpha_value
    Ainv_J_ground = solve_columns(ground_lu, J_ground) / alpha_value

    Dpsi = contact_dpsi_matrix(psi).toarray()
    schur = Dpsi - C_ball_weak @ Ainv_J_ball - C_ground_weak @ Ainv_J_ground
    rhs_psi = -rc - C_ball_weak @ base_ball - C_ground_weak @ base_ground
    try:
        dpsi = np.linalg.solve(schur, rhs_psi)
    except np.linalg.LinAlgError:
        dpsi = np.linalg.lstsq(schur, rhs_psi, rcond=None)[0]

    dub = base_ball - Ainv_J_ball @ dpsi
    dug = base_ground - Ainv_J_ground @ dpsi
    return np.concatenate((dub, dug, dpsi))


# Compute one Newton correction by solving the full sparse saddle system directly.
def contact_newton_step_direct(residual, psi, alpha_value, geometry):
    K = contact_sparse_linear_system(psi, alpha_value, geometry)
    try:
        return spla.spsolve(K, -residual)
    except RuntimeError:
        return spla.lsmr(K, -residual, atol=1.0e-12, btol=1.0e-12)[0]


# Compute one Newton correction using this script’s selected block strategy.
def contact_newton_step(residual, psi, alpha_value, geometry):
    if contact_linear_solver == "schur":
        return contact_newton_step_schur(residual, psi, alpha_value, geometry)
    return contact_newton_step_direct(residual, psi, alpha_value, geometry)


# Newton and LVPP

# Newton solves the nonlinear equations for a fixed alpha; LVPP updates lag until contact settles.

# Run damped Newton iterations for one fixed LVPP outer state.
def saddle_newton_solve(x0, rb0, rg0, alpha_value, lag):
    x = x0.copy()
    best_norm = np.inf
    best_x = x.copy()
    best_last = None
    for it in range(newton_max_it):
        ub, ug, psi = split_unknown(x)
        residual, gap, geometry = contact_residual(ub, ug, psi, rb0, rg0, alpha_value, lag)
        norm = float(np.linalg.norm(residual) / max(math.sqrt(residual.size), 1.0))
        gap_eq = weak_slack_error(gap, psi)
        last = (it + 1, norm, gap_eq, float(np.min(gap)))
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
            best_last = last
        if norm < newton_tol and gap_eq < max(newton_tol, 0.25 * gap_eq_tol):
            return x, last

        dx = contact_newton_step(residual, psi, alpha_value, geometry)
        damping = 1.0
        accepted = False
        for _ in range(newton_line_search_steps):
            trial = x + damping * dx
            tub, tug, tpsi = split_unknown(trial)
            trial_residual, trial_gap, _ = contact_residual(tub, tug, tpsi, rb0, rg0, alpha_value, lag)
            trial_norm = float(np.linalg.norm(trial_residual) / max(math.sqrt(trial_residual.size), 1.0))
            trial_gap_eq = weak_slack_error(trial_gap, tpsi)
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


# Return the LVPP proximal parameter for the current outer iteration.
def lvpp_alpha(k):
    return min(lvpp_alpha0 * lvpp_alpha_growth**k, lvpp_alpha_max)


# Measure the displacement change between two LVPP outer iterates.
def lvpp_increment(ub, ug, ub_prev, ug_prev):
    return math.sqrt(float(np.mean((ub - ub_prev) ** 2) + np.mean((ug - ug_prev) ** 2)))


# Run the LVPP outer loop while solving ball, ground, and contact together.
def monolithic_lvpp_solve(rb0, rg0, ub_initial, ug_initial):
    global psi_lag
    initialise_lvpp_lag(flat(u_old), flat(ground_u_old))
    x = np.concatenate((ub_initial, ug_initial, psi_lag.copy()))
    last = None
    for k in range(lvpp_max_it):
        alpha_value = lvpp_alpha(k)
        lag = psi_lag.copy()
        ub_prev, ug_prev, _ = split_unknown(x)
        x, newton_info = saddle_newton_solve(x, rb0, rg0, alpha_value, lag)
        ub, ug, psi = split_unknown(x)
        gap = gap_values(ub, ug)
        gap_eq = weak_slack_error(gap, psi)
        du_value = lvpp_increment(ub, ug, ub_prev, ug_prev)
        pressure = np.maximum(0.0, -(psi - lag) / alpha_value)
        active = int(np.count_nonzero(pressure > 1.0e-9))
        max_pressure = float(np.max(pressure)) if pressure.size else 0.0
        min_gap = float(np.min(gap))
        last = (k + 1, alpha_value, du_value, gap_eq, min_gap, active, max_pressure, newton_info)
        if du_value < lvpp_tol and gap_eq < gap_eq_tol and min_gap >= min_gap_tol:
            psi_lag = psi.copy()
            return x, last
        psi_lag = psi.copy()

    k, alpha_value, du_value, gap_eq, min_gap, active, max_pressure, newton_info = last
    raise RuntimeError(
        "LVPP monolithic step did not converge: "
        f"it={k}, du={du_value:.3e}, gap_eq={gap_eq:.3e}, "
        f"min_gap={min_gap:.3e}, newton={newton_info}"
    )


# state update

# Once a step is accepted, update displacement, velocity, and acceleration for ball and ground.

# Accept the ball displacement and update its Newmark velocity and acceleration.
def accept_ball(ub):
    set_flat(u_state, ub)
    ball_a_new.interpolate(ball_accel_expr(u_state))
    ball_v_new.interpolate(ball_vel_expr(ball_a_new))
    u_old.assign(u_state)
    v_old.assign(ball_v_new)
    a_old.assign(ball_a_new)


# Accept the ground displacement and update its Newmark velocity and acceleration.
def accept_ground(ug):
    set_flat(ground_state, ug)
    ground_a_new.interpolate(ground_accel_expr(ground_state))
    ground_v_new.interpolate(ground_vel_expr(ground_a_new))
    ground_u_old.assign(ground_state)
    ground_v_old.assign(ground_v_new)
    ground_a_old.assign(ground_a_new)


# Recompute the smallest contact gap from the current accepted state.
def current_min_gap():
    return float(np.min(gap_values(flat(u_old), flat(ground_u_old))))


# Linearly blend two flat displacement vectors.
def blend_vector(a, b, theta):
    return a + theta * (b - a)


# Backtrack from free flight to find a positive-gap coupled entry state for LVPP.
def positive_lvpp_entry_state(ub_current, ug_current, ub_free, ug_free):
    free_gap = float(np.min(gap_values(ub_free, ug_free)))
    if free_gap > entry_gap_target:
        return ub_free, ug_free, "free_positive", 1.0, free_gap

    current_gap = float(np.min(gap_values(ub_current, ug_current)))
    if current_gap <= entry_gap_target:
        return (
            ub_current.copy(),
            ug_current.copy(),
            "current_near_contact",
            0.0,
            current_gap,
        )

    lo = 0.0
    hi = 1.0
    for _ in range(entry_backtrack_iters):
        mid = 0.5 * (lo + hi)
        ub_mid = blend_vector(ub_current, ub_free, mid)
        ug_mid = blend_vector(ug_current, ug_free, mid)
        mid_gap = float(np.min(gap_values(ub_mid, ug_mid)))
        if mid_gap > entry_gap_target:
            lo = mid
        else:
            hi = mid

    theta = float(np.clip(lo * entry_theta_safety, 0.0, 1.0))
    ub_entry = blend_vector(ub_current, ub_free, theta)
    ug_entry = blend_vector(ug_current, ug_free, theta)
    entry_gap = float(np.min(gap_values(ub_entry, ug_entry)))
    return ub_entry, ug_entry, "backtrack", theta, entry_gap


# rendering

# Rendering only draws the current deformation: ball surface, ground top surface, and edges.

frames = []
ball_coords0 = np.asarray(ball_mesh.coordinates.dat.data_ro, dtype=float).copy()
ground_coords0 = np.asarray(ground_mesh.coordinates.dat.data_ro, dtype=float).copy()
u_ball_plot = Function(ball_mesh.coordinates.function_space(), name="u_ball_plot")
u_ground_plot = Function(ground_mesh.coordinates.function_space(), name="u_ground_plot")
ball_triangles = make_mesh_surface_triangles(ball_mesh)
ground_triangles = make_mesh_surface_triangles(ground_mesh)
ground_edges = unique_triangle_edges(ground_triangles)
top_edge_mask = np.all(np.isclose(ground_coords0[ground_edges, 1], ground_y), axis=1)
ground_top_edges = ground_edges[top_edge_mask]

x_min = ground_x0 - 0.12
x_max = ground_x1 + 0.12
z_min = ground_z0 - 0.12
z_max = ground_z1 + 0.12
y_min = ground_y - ground_depth - 0.05
y_max = cy + 0.85 * radius


# Render the current simulation state into the video frame buffer.
def render_frame():
    if not write_video:
        return

    u_ball_plot.interpolate(u_old)
    u_ground_plot.interpolate(ground_u_old)
    ball_coords = ball_coords0 + np.asarray(u_ball_plot.dat.data_ro, dtype=float)
    ground_coords = ground_coords0 + np.asarray(u_ground_plot.dat.data_ro, dtype=float)
    ball_faces = ball_coords[:, [0, 2, 1]][ball_triangles]
    ground_faces = ground_coords[:, [0, 2, 1]][ground_triangles]

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            ground_faces,
            facecolor=(0.70, 0.70, 0.67, 0.22),
            edgecolor=(0.10, 0.10, 0.10, 0.32),
            linewidth=0.18,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            ball_faces,
            facecolor=(0.25, 0.52, 0.78, 0.30),
            edgecolor=(0.10, 0.28, 0.46, 0.52),
            linewidth=0.22,
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            ground_coords[:, [0, 2, 1]][ground_top_edges],
            colors=(0.08, 0.08, 0.08, 0.42),
            linewidths=0.42,
        )
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(y_min, y_max)
    ax.set_box_aspect((x_max - x_min, z_max - z_min, y_max - y_min))
    ax.view_init(elev=14.0, azim=-52.0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)


# time loop

# Main loop: free predictor -> entry backtrack -> monolithic contact solve -> accept.

print(
    "3D deformable-ground monolithic LVPP | "
    f"Vb={V.dim()} Vg={G.dim()} Q={Q.dim()} total={V.dim() + G.dim() + Q.dim()} | "
    f"dt={dt_value:.2e} steps={num_steps} | Eb={E_ball_value:.2e} Eg={E_ground_value:.2e} | "
    f"lvpp_tol={lvpp_tol:.1e} min_gap_tol={min_gap_tol:.1e} | "
    f"normal=Piola F^-T N | linear_solver={contact_linear_solver} | "
    f"contact_space={contact_space_family}{contact_space_degree} | "
    f"entry_gap_target={entry_gap_target:.1e} | "
    f"entry_backtrack_iters={entry_backtrack_iters} | "
    f"entry_theta_safety={entry_theta_safety:.3f} | "
    f"maxh={maxh:.3g} contact_maxh={contact_maxh:.3g} | "
    f"ground={ground_nx}x{ground_ny}x{ground_nz}"
)

simulation_start = time.perf_counter()
contact_steps = 0
max_lvpp_iters = 0
global_min_gap = current_min_gap()
accepted_gap_violations = 0
min_trial_gap = math.inf

render_frame()

for step in range(1, num_steps + 1):
    rb0 = zero_ball_residual()
    rg0 = zero_ground_residual()
    ub_free = ball_lu(-rb0)
    ug_free = ground_lu(-rg0)
    trial_gap = float(np.min(gap_values(ub_free, ug_free)))
    min_trial_gap = min(min_trial_gap, trial_gap)
    lvpp_info = None

    if trial_gap > activation_gap:
        accept_ball(ub_free)
        accept_ground(ug_free)
        min_gap = float(np.min(gap_values(ub_free, ug_free)))
    else:
        contact_steps += 1
        ub_current = flat(u_old).copy()
        ug_current = flat(ground_u_old).copy()
        ub_initial, ug_initial, _, _, _ = positive_lvpp_entry_state(
            ub_current,
            ug_current,
            ub_free,
            ug_free,
        )
        x, lvpp_info = monolithic_lvpp_solve(rb0, rg0, ub_initial, ug_initial)
        ub, ug, _ = split_unknown(x)
        accept_ball(ub)
        accept_ground(ug)
        min_gap = float(np.min(gap_values(ub, ug)))
        max_lvpp_iters = max(max_lvpp_iters, lvpp_info[0])

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
