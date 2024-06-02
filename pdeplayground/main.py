import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from jax import lax  # Import the lax module


# Parameters
nx, ny = 100, 100  # Grid size
dx, dy = 1.0, 1.0  # Grid spacing
c = 1.0            # Wave speed
dt = 0.1           # Time step
nt = 100           # Number of time steps

# Initial conditions
u = jnp.zeros((nx, ny))      # u at time n
u_new = jnp.zeros((nx, ny))  # u at time n+1
u_old = jnp.zeros((nx, ny))  # u at time n-1
u = u.at[nx//2, ny//2].set(1.0)  # Initial disturbance

# Laplacian operator for a 2D grid with Dirichlet boundary conditions
def laplacian_2d(u, dx, dy):
    nx, ny = u.shape
    N = nx * ny
    dx2_inv = 1 / dx**2
    dy2_inv = 1 / dy**2

    def get_indices_data(i, j):
        
        idx = i * ny + j
        indices = [
            (idx, idx, -2 * (dx2_inv + dy2_inv)),
            (idx, lax.cond(i > 0, lambda _: idx - ny, lambda _: idx, None), lax.cond(i > 0, lambda _: dx2_inv, lambda _: 0, None)),
            (idx, lax.cond(i < nx - 1, lambda _: idx + ny, lambda _: idx, None), lax.cond(i < nx - 1, lambda _: dx2_inv, lambda _: 0, None)),
            (idx, lax.cond(j > 0, lambda _: idx - 1, lambda _: idx, None), lax.cond(j > 0, lambda _: dy2_inv, lambda _: 0, None)),
            (idx, lax.cond(j < ny - 1, lambda _: idx + 1, lambda _: idx, None), lax.cond(j < ny - 1, lambda _: dy2_inv, lambda _: 0, None))
        ]
        return jnp.array(indices)

    grid_i, grid_j = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
    grid_indices = jnp.stack((grid_i.flatten(), grid_j.flatten()), axis=-1)

    indices_data = jax.vmap(lambda ij: get_indices_data(ij[0], ij[1]))(grid_indices)
    indices_data = indices_data.reshape(-1, 3)

    row_indices = indices_data[:, 0].astype(int)
    col_indices = indices_data[:, 1].astype(int)
    data = indices_data[:, 2]

    laplacian_matrix = jnp.zeros((N, N))
    laplacian_matrix = laplacian_matrix.at[(row_indices, col_indices)].add(data)
    return laplacian_matrix


# Explicit time-stepping function
def explicit_step(u_new, u, u_old, c, dt, dx, dy):
    coeff = (c * dt / dx) ** 2
    u_new = (2 * u - u_old +
             coeff * (jax.lax.fori_loop(1, nx-1, lambda i, acc: acc.at[i, 1:-1].set(
                 u[i+1, 1:-1] + u[i-1, 1:-1] + u[i, 2:] + u[i, :-2] - 4*u[i, 1:-1]
             ), u_new)))
    return u_new

# Implicit time-stepping function
def implicit_step(u_new, u, u_old, c, dt, dx, dy):
    alpha = (c * dt / dx) ** 2
    I = jnp.eye(nx * ny)
    L = laplacian_2d(u, dx, dy)
    A = I + 0.5 * alpha * L
    B = 2 * u - u_old - 0.5 * alpha * L
    u_new = jax.scipy.sparse.linalg.cg(A, B.reshape(-1))[0].reshape((nx, ny))
    return u_new

# Main function to run the simulation
def run_simulation(method='explicit'):
    global u, u_new, u_old
    for n in range(nt):
        if method == 'explicit':
            u_new = explicit_step(u_new, u, u_old, c, dt, dx, dy)
        elif method == 'implicit':
            u_new = implicit_step(u_new, u, u_old, c, dt, dx, dy)
        u_old, u, u_new = u, u_new, u_old  # Rotate references
    return u


def test_laplacian():
    # Test grid
    U = jnp.array([
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ])
    dx, dy = 1.0, 1.0  # Example grid spacing

    # Construct the Laplacian matrix
    L = laplacian_2d(U, dx, dy)

    # Flatten U
    U_flat = U.flatten()

    # Apply Laplacian matrix
    laplacian_U = L @ U_flat
    
    # Manually compute the Laplacian
    U_laplacian_manual = jnp.array([
        [0, 0, 0],
        [0, -4, 0],
        [0, 0, 0]
    ]).flatten()

    # Check if the results are close
    assert jnp.allclose(laplacian_U, U_laplacian_manual), f"Computed Laplacian: {laplacian_U}, Expected: {U_laplacian_manual}"
    print("Laplacian test passed!")

# Run the test
test_laplacian()

# Compare both methods
#u_explicit = run_simulation(method='explicit')
#u_implicit = run_simulation(method='implicit')

#print("Explicit Method Result:\n", u_explicit)
#print("Implicit Method Result:\n", u_implicit)