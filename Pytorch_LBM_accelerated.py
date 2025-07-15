from __future__ import annotations
import torch
import matplotlib.pyplot as plt


def get_memory_usage():
    """Get current GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2     # MB
        return f"GPU: {allocated:.1f}MB allocated, {cached:.1f}MB cached"
    else:
        return "CPU mode"

# ────────────────────────── Configuration ────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nx, Ny = 3000, 3000               # grid resolution (x, y) 
tau = 0.6                        # relaxation time 
u_lid = 0.05                     # lid velocity (lattice units)
max_iter = 100               # hard iteration cap
check_it = 100                  # test convergence every N iters - reduced frequency for performance
viz_every = 50                 # live‑plot refresh interval
rho0 = 1.0                      # initial density

# Visualization control
enable_plotting = False          # Set to False to disable all plotting for performance
show_final_plot = False          # Set to False to disable final streamlines plot

# Performance monitoring
perf_every = 10000               # BLUPS calculation interval - reduced frequency for performance

#time 
import time

# ───────────────────────── Lattice constants ─────────────────────────
c = torch.tensor([[ 0, 0], [ 1, 0], [ 0, 1], [-1, 0], [ 0,-1],
                  [ 1, 1], [-1, 1], [-1,-1], [ 1,-1]], dtype=torch.float32, device=DEVICE)  # (9,2)
w = torch.tensor([4/9] + [1/9]*4 + [1/36]*4, dtype=torch.float32, device=DEVICE)            # (9,)
# Split components for dot‑products
cx, cy = c[:, 0], c[:, 1]

# ────────────────────────── Helper functions ─────────────────────────

def feq(rho: torch.Tensor, u: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """D2Q9 equilibrium distribution."""
    cu  = u[..., 0, None] * cx + u[..., 1, None] * cy
    usq = (u**2).sum(-1, keepdim=True)
    result = rho[..., None] * w * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
    if out is not None:
        out.copy_(result)
        return out
    return result

#@torch.compile
def lbm_step(f: torch.Tensor, f_temp: torch.Tensor, feq_temp: torch.Tensor, 
             rho: torch.Tensor, u: torch.Tensor, u_lid_tensor: torch.Tensor,
             cx: torch.Tensor, cy: torch.Tensor, tau: float) -> None:
    """Complete LBM timestep with torch.compile optimization - all operations fused."""
    
    # Collision (BGK) - reuse feq_temp buffer
    cu = u[..., 0, None] * cx + u[..., 1, None] * cy
    usq = (u**2).sum(-1, keepdim=True)
    feq_temp.copy_(rho[..., None] * w * (1 + 3*cu + 4.5*cu**2 - 1.5*usq))
    f += (feq_temp - f) / tau

    # Streaming - use original torch.roll (optimized by torch.compile)
    f_temp[..., 0] = f[..., 0]
    f_temp[..., 1] = torch.roll(f[..., 1], shifts=( 0,  1), dims=(0, 1))
    f_temp[..., 2] = torch.roll(f[..., 2], shifts=(-1,  0), dims=(0, 1))
    f_temp[..., 3] = torch.roll(f[..., 3], shifts=( 0, -1), dims=(0, 1))
    f_temp[..., 4] = torch.roll(f[..., 4], shifts=( 1,  0), dims=(0, 1))
    f_temp[..., 5] = torch.roll(f[..., 5], shifts=(-1,  1), dims=(0, 1))
    f_temp[..., 6] = torch.roll(f[..., 6], shifts=(-1, -1), dims=(0, 1))
    f_temp[..., 7] = torch.roll(f[..., 7], shifts=( 1, -1), dims=(0, 1))
    f_temp[..., 8] = torch.roll(f[..., 8], shifts=( 1,  1), dims=(0, 1))
    
    # Swap buffers (in-place copy)
    f.copy_(f_temp)

    # Boundaries - bounce back (optimized)
    # Left wall (x = 0): swap 1↔3, 5↔6, 8↔7
    f[:, 0, [1, 5, 8]], f[:, 0, [3, 6, 7]] = f[:, 0, [3, 6, 7]], f[:, 0, [1, 5, 8]]
    # Right wall (x = Nx‑1): swap 3↔1, 6↔5, 7↔8
    f[:, -1, [3, 6, 7]], f[:, -1, [1, 5, 8]] = f[:, -1, [1, 5, 8]], f[:, -1, [3, 6, 7]]
    # Bottom wall (y = 0): swap 2↔4, 5↔7, 6↔8
    f[0, :, [2, 5, 6]], f[0, :, [4, 7, 8]] = f[0, :, [4, 7, 8]], f[0, :, [2, 5, 6]]

    # Moving lid boundary (top wall)
    rho_top = (f[-1, :, [0, 1, 3, 2, 5, 6]].sum(-1) + 2*f[-1, :, 4])
    cu_top = u_lid_tensor[0] * cx + u_lid_tensor[1] * cy
    usq_top = (u_lid_tensor**2).sum()
    fe_top = rho_top[..., None] * w * (1 + 3*cu_top + 4.5*cu_top**2 - 1.5*usq_top)
    f[-1, :, 2] = fe_top[:, 2] + f[-1, :, 4] - fe_top[:, 4]
    f[-1, :, 5] = fe_top[:, 5] + f[-1, :, 7] - fe_top[:, 7]
    f[-1, :, 6] = fe_top[:, 6] + f[-1, :, 8] - fe_top[:, 8]

    # Macroscopic fields - compute in-place
    torch.sum(f, dim=-1, out=rho)
    torch.sum(f * cx, dim=-1, out=u[..., 0])
    torch.sum(f * cy, dim=-1, out=u[..., 1])
    u[..., 0] /= rho
    u[..., 1] /= rho

    # Enforce no-slip conditions
    u[:, 0, :]  = 0.0  # left
    u[:, -1, :] = 0.0  # right
    u[0, :, :]  = 0.0  # bottom

# ────────────────────── Field initialisation ─────────────────────────
rho = torch.full((Ny, Nx), rho0, device=DEVICE)
u   = torch.zeros((Ny, Nx, 2), device=DEVICE)
f   = feq(rho, u).clone()

# Pre-allocate memory-efficient buffers
f_temp = torch.empty_like(f)  # for double-buffered streaming (avoids allocation)
feq_temp = torch.empty_like(f)  # for equilibrium calculations (reused buffer)
u_lid_tensor = torch.tensor([u_lid, 0.0], device=DEVICE)  # pre-allocated lid velocity

# Only allocate visualization buffers if plotting is enabled
if enable_plotting:
    speed_vis_cpu = torch.empty((Ny, Nx), dtype=torch.float32)  # CPU buffer for visualization


# Initialize performance monitoring
perf_start_time = time.time()
perf_start_iter = 0
total_lattice_updates = 0


# ───────────────────── Live visual initialisation ────────────────────
if enable_plotting:
    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(6, 5))
    img = ax_live.imshow(torch.zeros((Ny, Nx)), origin="lower", cmap="turbo", vmin=0, vmax=u_lid*1.2)
    fig_live.colorbar(img, ax=ax_live, label="|u| (lattice units)")
    ax_live.set_title("Velocity magnitude – live")
    fig_live.tight_layout()
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()



start_time = time.time()
# ───────────────────────────── Main loop ─────────────────────────────
it = 1
while it < max_iter + 1:
    # Complete LBM timestep with torch.compile optimization
    lbm_step(f, f_temp, feq_temp, rho, u, u_lid_tensor, cx, cy, tau)

    if it % perf_every == 0:
        # Calculate performance metrics every `perf_every` iterations
        perf_end_time = time.time()
        elapsed_time = perf_end_time - start_time
        total_lattice_updates = it * Nx * Ny
        million_lups = total_lattice_updates / (elapsed_time * 1e6)
        
        print(f"Iteration {it:>5}, Time: {elapsed_time:.2f}s, "
              f"MLUPS: {million_lups:.2f}, "
              f"Total updates: {total_lattice_updates:,}")
        

    # Live visual every `viz_every` steps
    if enable_plotting and it % viz_every == 0:
        speed_magnitude = u.norm(dim=-1)
        speed_vis_cpu.copy_(speed_magnitude)
        speed_vis_cpu = speed_vis_cpu.detach().cpu()
        img.set_data(speed_vis_cpu)
        img.set_clim(vmin=0, vmax=speed_vis_cpu.max().item())
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()
    it += 1

# Calculate final performance metrics
final_time = time.time()
total_runtime = final_time - start_time
final_million_lups = (max_iter*Nx*Ny) / (total_runtime*1e6)  # Million Lattice Updates Per Second
print(f"\n📊 Performance Summary:")
print(f"Total iterations: {it}")
print(f"Total lattice updates: {max_iter * Nx*Ny:,}")
print(f"Average MLUPS: {final_million_lups:.3f}")
print(f"Peak memory usage: {get_memory_usage()}")

print(f"Final memory usage: {get_memory_usage()}")
#PRINT END TIME
end_time = time.time()
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total runtime: {end_time - start_time:.2f} seconds")

# ───────────────── Refresh live figure one last time ─────────────────
if enable_plotting:
    final_speed = u.norm(dim=-1).detach().cpu()
    img.set_data(final_speed)
    img.set_clim(vmin=0, vmax=final_speed.max().item())
    ax_live.set_title("Velocity magnitude – final")
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()

# ───────────────────────── Final visualisations ──────────────────────
if show_final_plot:
    plt.ioff()
    print("Plotting final snapshot with streamlines…")

    final_speed = u.norm(dim=-1).detach().cpu() if not enable_plotting else final_speed
    speed = final_speed.numpy()
    step = 1  # full resolution for accurate streamlines
    Y, X = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing="ij")
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    U_np = u[..., 0].cpu().numpy()
    V_np = -u[..., 1].cpu().numpy()  # Flip sign: array row index increases upwards, but positive uy is north (row-), so negate

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(speed, origin="lower", cmap="turbo")
    ax.streamplot(X_np, Y_np, U_np, V_np, density=2.0, linewidth=0.8, arrowsize=1)
    ax.set_title("Velocity magnitude + streamlines (steady state)")
    plt.tight_layout()
    plt.show()
else:
    print("Final visualization disabled - calculation completed successfully")
