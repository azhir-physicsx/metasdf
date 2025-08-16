# %% [markdown]
# # Generating SDF Fields from MNIST and confirming validity

# %%
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as edt



# %%
# Load MNIST grayscale (no resize here)
mnist = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())

# %%
img28, label = mnist[0]                   # img28: [1,28,28] in [0,1]

# Resize FIRST (antialiased), then threshold
target_size = (128, 128)
resize = transforms.Resize(target_size, antialias=True)  # bilinear for PIL under the hood
img128 = resize(img28)                     # [1,128,128], still grayscale
# Threshold to create binary mask
threshold = 0.4
mask = (img128 >= threshold).float()       # binary mask [1,128,128]    


# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img28.squeeze(), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Upscaled Image")
plt.imshow(img128.squeeze(), cmap='gray')


plt.subplot(1, 3, 3)
plt.title("Generated Geometry")
plt.imshow(mask.squeeze(), cmap='gray')
plt.savefig("./figures/upscale_mnist.png", bbox_inches='tight', dpi=300)
plt.show()

# %% [markdown]
# # Calculating the SDF Field

# %%
mask = (mask.squeeze().detach().cpu().numpy()).astype(np.uint8)

# Distances: edt computes, for each ZERO pixel, distance to nearest NON-ZERO pixel
dist_outside = edt(mask == 0)  # distance for background pixels to the digit
dist_inside  = edt(mask == 1)  # distance for foreground pixels to the background

sdf = dist_outside - dist_inside          # NEGATIVE inside (mask==1), POSITIVE outside

# Optional: match MetaSDF scale (divide by image width) - sdf /= float(mask.shape[1])

# %% [markdown]
# # Solving PDEs

# %%
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def assemble_diffusion_fullbox(k, f, kappa2=None, spacing=(1.0,1.0),
                               bc='neumann', g_box=0.0):
    """
    Assemble A u = b for: -div(k grad u) + kappa2*u = f on the full HxW box.

    Parameters
    ----------
    k : (H,W) array            conductivity per cell (>=0)
    f : (H,W) array            source term
    kappa2 : (H,W) array or None   screening term (>=0); if None, zeros
    spacing : (dy,dx)
    bc : 'neumann' or 'dirichlet'  box boundary condition
    g_box : float or (H,W) array   Dirichlet value if bc='dirichlet'

    Notes
    -----
    - Uses 5-point stencil in divergence form with *harmonic* face conductivities.
    - Neumann: zero normal flux at box edges (one-sided stencil).
    - Dirichlet: fix boundary nodes to g_box.
    """
    k = np.asarray(k, float); f = np.asarray(f, float)
    H, W = k.shape
    if kappa2 is None: kappa2 = np.zeros_like(k)
    else: kappa2 = np.asarray(kappa2, float)
    dy, dx = spacing
    inv_dx2, inv_dy2 = 1.0/(dx*dx), 1.0/(dy*dy)

    N = H*W
    def idx(i,j): return i*W + j

    rows, cols, data = [], [], []
    b = np.zeros(N, float)

    # Precompute neighbor indices
    for i in range(H):
        for j in range(W):
            p = idx(i,j)
            if bc=='dirichlet' and (i==0 or i==H-1 or j==0 or j==W-1):
                # u_p = g_box
                rows.append(p); cols.append(p); data.append(1.0)
                b[p] = g_box if np.isscalar(g_box) else float(g_box[i,j])
                continue

            diag = 0.0

            # West (i,j-1) ←→ (i,j)
            if j > 0:
                q = idx(i, j-1)
                k_face = 2.0*k[i,j]*k[i,j-1] / (k[i,j]+k[i,j-1]+1e-12)
                w = k_face * inv_dx2
                rows += [p, p,]; cols += [p, q]; data += [ w, -w]
                diag += w
            else:
                if bc=='neumann':
                    # zero-flux: no neighbor, nothing to add
                    pass

            # East (i,j+1)
            if j < W-1:
                q = idx(i, j+1)
                k_face = 2.0*k[i,j]*k[i,j+1] / (k[i,j]+k[i,j+1]+1e-12)
                w = k_face * inv_dx2
                rows += [p, p,]; cols += [p, q]; data += [ w, -w]
                diag += w
            else:
                if bc=='neumann':
                    pass

            # South (i-1,j)
            if i > 0:
                q = idx(i-1, j)
                k_face = 2.0*k[i,j]*k[i-1,j] / (k[i,j]+k[i-1,j]+1e-12)
                w = k_face * inv_dy2
                rows += [p, p,]; cols += [p, q]; data += [ w, -w]
                diag += w
            else:
                if bc=='neumann':
                    pass

            # North (i+1,j)
            if i < H-1:
                q = idx(i+1, j)
                k_face = 2.0*k[i,j]*k[i+1,j] / (k[i,j]+k[i+1,j]+1e-12)
                w = k_face * inv_dy2
                rows += [p, p,]; cols += [p, q]; data += [ w, -w]
                diag += w
            else:
                if bc=='neumann':
                    pass

            # Diagonal (accumulated) + screening
            rows.append(p); cols.append(p); data.append(diag + kappa2[i,j])

            # RHS
            b[p] = f[i,j]

    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A, b


# %%
inside = sdf < 0

# Strong contrast
k_in, k_out = 1.0, 0.05
k = np.where(inside, k_in, k_out)

# Source only inside
f_in = 1.0
f = np.where(inside, f_in, 0.0)

# Add decay outside to make geometry "glow"
kappa2 = np.where(inside, 0.0, 0.02)

A, b = assemble_diffusion_fullbox(
    k, f, kappa2=kappa2,
    spacing=(1.0,1.0),
    bc='neumann'   # keep zero-flux on box edges
)
u = spsolve(A, b).reshape(sdf.shape)


# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(u, origin='lower', aspect='equal')
ax.contour(sdf, levels=[0.0], colors='k', linewidths=1.0)
fig.colorbar(im, ax=ax); ax.set_title("Two-phase diffusion (whole box)")
plt.tight_layout(); plt.show()



