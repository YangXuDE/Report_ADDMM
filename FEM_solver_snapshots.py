import gmsh
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from dolfinx.io import gmshio
from dolfinx.fem import locate_dofs_topological
import os

# Step 1: Generate mesh using Gmsh (带孔洞版本)
def generate_mesh(mesh_file="tensile_test_specimen.msh"):
    gmsh.initialize()
    gmsh.model.add("Tensile Test Specimen")

    # Define dimensions
    L_t = 5
    L_roundx = 15
    L_specimen = 80
    W_fix = 30
    W_specimen = 20
    t = 4
    hole_center = (L_roundx + L_t + L_specimen/2, 0)
    hole_radius = 4
    transition_radius = 25

    # Define points
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(0, -W_fix/2, 0)
    p3 = gmsh.model.geo.addPoint(L_roundx, -W_specimen/2, 0)
    p4 = gmsh.model.geo.addPoint(L_roundx + L_t, -W_specimen/2, 0)
    p5 = gmsh.model.geo.addPoint(L_roundx + L_t + L_specimen, -W_specimen/2, 0)
    p6 = gmsh.model.geo.addPoint(L_roundx + L_t + L_specimen, W_specimen/2, 0)
    p7 = gmsh.model.geo.addPoint(L_roundx + L_t, W_specimen/2, 0)
    p8 = gmsh.model.geo.addPoint(L_roundx, W_specimen/2, 0)
    p9 = gmsh.model.geo.addPoint(0, W_fix/2, 0)
    p10 = gmsh.model.geo.addPoint(hole_center[0], hole_center[1], 0)
    cp2 = gmsh.model.geo.addPoint(L_roundx, transition_radius * 4/5 + W_fix/2, 0)
    cp1 = gmsh.model.geo.addPoint(L_roundx, -(transition_radius * 4/5 + W_fix/2), 0)

    # Define lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2arc = gmsh.model.geo.addCircleArc(p2, cp1, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8arc = gmsh.model.geo.addCircleArc(p8, cp2, p9)
    l9 = gmsh.model.geo.addLine(p9, p1)

    # Hole definition
    p_hole1 = gmsh.model.geo.addPoint(hole_center[0] + hole_radius, hole_center[1], 0)
    p_hole2 = gmsh.model.geo.addPoint(hole_center[0] - hole_radius, hole_center[1], 0)
    arc1 = gmsh.model.geo.addCircleArc(p_hole1, p10, p_hole2)
    arc2 = gmsh.model.geo.addCircleArc(p_hole2, p10, p_hole1)

    # Define curve loops and surface (带孔洞)
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2arc, l3, l4, l5, l6, l7, l8arc, l9])
    hole_loop = gmsh.model.geo.addCurveLoop([arc1, arc2])
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [surface], tag=1)  # Surface
    gmsh.model.addPhysicalGroup(1, [arc1, arc2], tag=2)  # Hole boundary
    gmsh.model.addPhysicalGroup(1, [l1, l9], tag=3)  # Dirichlet boundary
    gmsh.model.addPhysicalGroup(1, [l5], tag=4)  # Neumann boundary

    # Mesh generation
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file)
    gmsh.finalize()
    return mesh_file

# Step 2: FEM functions from DOLFINx
def generate_snapshots(domain, facet_tags, num_snapshots=300, G_range=(50e9, 90e9), K_range=(100e9, 150e9), save_folder=None, subdomain_condition=None):
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "snapshots")
    os.makedirs(save_folder, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for i in range(rank, num_snapshots, size):
        G = np.random.uniform(G_range[0], G_range[1])
        K = np.random.uniform(K_range[0], K_range[1])
        FEM_sol(domain, facet_tags, G, K, save_folder=save_folder, subdomain_condition=subdomain_condition)
    comm.Barrier()

def generate_test_snapshot(domain, facet_tags, G, K, save_folder=None, subdomain_condition=None):
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "test_snapshot")
    os.makedirs(save_folder, exist_ok=True)
    FEM_sol(domain, facet_tags, G, K, save_folder=save_folder, subdomain_condition=subdomain_condition)

def FEM_sol(domain, facet_tags, G, K, T_neuman=None, u_D=None, dirichlet_tag=3, neumann_tag=4, save_folder=None, subdomain_condition=None):
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    # Dirichlet BC
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    b_D = locate_dofs_topological(V, fdim, facet_tags.find(dirichlet_tag))
    if u_D is None:
        u_D = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
    bc_D = fem.dirichletbc(u_D, b_D, V)

    # Neumann BC
    if T_neuman is None:
        T_neuman = fem.Constant(domain, default_scalar_type((106.26e6, 0, 0)))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # Weak form
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u):
        return (-2 / 3 * G + K) * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * G * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T_neuman, v) * ds(neumann_tag)

    # Solve
    problem = LinearProblem(a, L, bcs=[bc_D], petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
    uh = problem.solve()

    # Extract subdomain
    domain_arr = domain.geometry.x
    if subdomain_condition is None:
        x_min, x_max = np.min(domain_arr[:, 0]), np.max(domain_arr[:, 0])
        x_mid = (x_min + x_max) / 2
        x_range = (x_max - x_min) * 0.25
        subdomain_condition = lambda x: (x[:, 0] >= x_mid - x_range) & (x[:, 0] <= x_mid + x_range)
    
    condition = subdomain_condition(domain_arr)
    domain_sub = domain_arr[condition].copy()
    domain_sub[:, 0] -= np.min(domain_sub[:, 0])
    domain_sub[:, 1] += 10

    num_dofs = V.dofmap.index_map.size_local
    uh_arr = uh.x.array[:].reshape((num_dofs, -1))
    u_sub = uh_arr[condition]

    # Save snapshot with header
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        filename = os.path.join(save_folder, f"snapshot_G{G:.2e}_K{K:.2e}.csv")
        data = np.column_stack((domain_sub[:, 0], domain_sub[:, 1], u_sub[:, 0], u_sub[:, 1]))
        header = "x-coordinate [mm],y-coordinate [mm],x-displacement [mm],y-displacement [mm]"
        np.savetxt(filename, data, delimiter=",", header=header, comments="")

    return domain, uh, domain_sub, u_sub

def load_mesh(mesh_file):
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file '{mesh_file}' not found.")
    domain, _, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0)
    return domain, facet_tags

# Main execution
if __name__ == "__main__":
    # Generate and load mesh
    mesh_file = generate_mesh()
    domain, facet_tags = load_mesh(mesh_file)

    # Custom subdomain condition
    custom_subdomain = lambda x: (x[:, 0] >= 20) & (x[:, 0] <= 100)

    # Generate test snapshot
    generate_test_snapshot(domain, facet_tags, G=7.35e10, K=1.28e11, subdomain_condition=custom_subdomain)

    # Generate training snapshots
    generate_snapshots(domain, facet_tags, num_snapshots=20, G_range=(5e10, 1e11), K_range=(1e11, 1.5e11), subdomain_condition=custom_subdomain)

    # # Custom FEM solve
    # G, K = 100e9, 100e9
    # custom_T_neuman = fem.Constant(domain, default_scalar_type((50e6, 0, 0)))
    # custom_u_D = np.array([0.0, 0.1, 0.0], dtype=default_scalar_type)
    # domain, uh, domain_sub, u_sub = FEM_sol(domain, facet_tags, G, K, T_neuman=custom_T_neuman, u_D=custom_u_D, subdomain_condition=custom_subdomain)