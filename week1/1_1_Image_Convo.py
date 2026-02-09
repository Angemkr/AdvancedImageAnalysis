import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.io


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_gaussian_kernel(sigma, truncate=4):
    """
    Create a 1D Gaussian kernel, truncated at `truncate` * sigma
    and normalized so that all values sum to 1.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    truncate : float
        Number of standard deviations at which to truncate.

    Returns
    -------
    x : ndarray (1D)
        Integer positions from -radius to +radius.
    kernel : ndarray (1D)
        Normalized Gaussian kernel.
    """
    radius = int(np.ceil(sigma * truncate))
    x = np.arange(-radius, radius + 1)

    # Gaussian formula (eq 1.4), unnormalized first
    kernel = np.exp(-x**2 / (2 * sigma**2))

    # Normalize to sum=1 (better than analytic constant
    # because the kernel is truncated)
    kernel = kernel / kernel.sum()

    return x, kernel


def create_gaussian_derivative_kernel(sigma, truncate=4):
    """
    Create a 1D Gaussian derivative kernel using the analytic
    formula dg/dx = -(x / t) * g(x) where t = sigma^2 (eq 1.8).

    Built on the normalized Gaussian so that magnitudes are
    consistent with create_gaussian_kernel.

    Returns
    -------
    x : ndarray (1D)
        Integer positions.
    kernel : ndarray (1D)
        Gaussian derivative kernel (sums to ~0).
    """
    # Start from the normalized Gaussian
    x, g = create_gaussian_kernel(sigma, truncate)

    # Analytic derivative: multiply by -(x / t)
    t = sigma**2
    kernel = -(x / t) * g

    return x, kernel


# =============================================================================
# TASK 1: Create and visualize Gaussian kernels
# =============================================================================

print("=" * 60)
print("TASK 1: Create Gaussian Kernel")
print("=" * 60)

sigma = 4.5
x, g = create_gaussian_kernel(sigma)
x_d, g_d = create_gaussian_derivative_kernel(sigma)

# Sanity checks
print(f"  sigma           = {sigma}")
print(f"  kernel radius   = {len(x) // 2}")
print(f"  kernel sum      = {g.sum():.10f}  (expect 1.0)")
print(f"  deriv kernel sum= {g_d.sum():.10f}  (expect ~0.0)")
print()

# --- Plot Gaussian (Figure 1.4) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, g, "b-", linewidth=2)
axes[0].set_title(f"Gaussian (σ = {sigma})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("g(x)")
axes[0].grid(True, alpha=0.3)

# --- Plot Gaussian derivative (Figure 1.5) ---
axes[1].plot(x_d, g_d, "r-", linewidth=2)
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].set_title(f"Gaussian derivative (σ = {sigma})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("g'(x)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# LOAD TEST IMAGE
# =============================================================================

filename = "week1/week1_data/fibres_xcth.png"  # adjust path as needed
im = skimage.io.imread(filename).astype(float)

print(f"Image loaded: shape={im.shape}, dtype={im.dtype}")
print()


# =============================================================================
# TASK 2: Verify separability of the Gaussian
#
# Idea: convolving with one 2D kernel (outer product) should
#       give the same result as two sequential 1D convolutions.
# =============================================================================

print("=" * 60)
print("TASK 2: Verify Separability")
print("=" * 60)

_, g_1d = create_gaussian_kernel(sigma)

# --- Method A: single 2D kernel (outer product of 1D × 1D) ---
g_2d = np.outer(g_1d, g_1d)
result_2d = scipy.ndimage.convolve(im, g_2d, mode="nearest")

# --- Method B: two 1D passes (horizontal then vertical) ---
result_sep = scipy.ndimage.convolve(
    im, g_1d.reshape(1, -1), mode="nearest"
)
result_sep = scipy.ndimage.convolve(
    result_sep, g_1d.reshape(-1, 1), mode="nearest"
)

# --- Compare ---
diff = result_2d - result_sep
print(f"  Mean |diff| = {np.mean(np.abs(diff)):.2e}")
print(f"  Max  |diff| = {np.max(np.abs(diff)):.2e}")
print("  (should be ~1e-10 or smaller — machine precision)")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(result_2d, cmap="gray")
axes[0].set_title("2D kernel")
axes[1].imshow(result_sep, cmap="gray")
axes[1].set_title("Separable (two 1D)")
im_d = axes[2].imshow(diff, cmap="bwr")
axes[2].set_title("Difference")
plt.colorbar(im_d, ax=axes[2])
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()


# =============================================================================
# TASK 3: Verify derivative property (eq 1.7)
#
# d/dx (I * g) = I * (dg/dx)
#
# Method A: smooth with Gaussian, then finite-difference [0.5, 0, -0.5]
# Method B: convolve directly with the Gaussian derivative kernel
#
# We test in 1D (horizontal direction) as suggested.
# =============================================================================

print("=" * 60)
print("TASK 3: Verify Derivative Property")
print("=" * 60)

# Method A: smooth first, then approximate derivative
im_smooth = scipy.ndimage.convolve(
    im, g_1d.reshape(1, -1), mode="nearest"
)
deriv_fd = np.array([[0.5, 0, -0.5]])  # central finite difference
result_A = scipy.ndimage.convolve(im_smooth, deriv_fd, mode="nearest")

# Method B: convolve with Gaussian derivative directly
_, g_d = create_gaussian_derivative_kernel(sigma)
result_B = scipy.ndimage.convolve(
    im, g_d.reshape(1, -1), mode="nearest"
)

# Compare — expect small but non-zero difference because [0.5,0,-0.5]
# is only an approximation of the true derivative
diff = result_A - result_B
print(f"  Mean |diff| = {np.mean(np.abs(diff)):.2e}")
print(f"  Max  |diff| = {np.max(np.abs(diff)):.2e}")
print("  (small but not machine-zero — finite difference is approximate)")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(result_A, cmap="RdBu_r")
axes[0].set_title("Smooth → finite diff")
axes[1].imshow(result_B, cmap="RdBu_r")
axes[1].set_title("Gaussian derivative")
im_d = axes[2].imshow(diff, cmap="bwr")
axes[2].set_title("Difference")
plt.colorbar(im_d, ax=axes[2])
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()


# =============================================================================
# TASK 4: Verify semi-group property (eq 1.6)
#
# g(t1+t2) * I  =  g(t1) * g(t2) * I
#
# Test: one convolution with t=20 vs. ten convolutions with t=2
# (10 × 2 = 20)
# =============================================================================

print("=" * 60)
print("TASK 4: Verify Semi-Group (Gaussian)")
print("=" * 60)

t_big = 20
t_small = 2

# --- Method A: single large Gaussian ---
_, g_big = create_gaussian_kernel(np.sqrt(t_big))
g_big_2d = np.outer(g_big, g_big)
result_A = scipy.ndimage.convolve(im, g_big_2d, mode="nearest")

# --- Method B: 10 sequential small Gaussians ---
_, g_small = create_gaussian_kernel(np.sqrt(t_small))
g_small_2d = np.outer(g_small, g_small)

result_B = im.copy()
for _ in range(10):
    result_B = scipy.ndimage.convolve(
        result_B, g_small_2d, mode="nearest"
    )

# Compare
diff = result_A - result_B
print(f"  t_big={t_big} (σ={np.sqrt(t_big):.3f})")
print(f"  t_small={t_small} (σ={np.sqrt(t_small):.3f}), 10 iterations")
print(f"  Mean |diff| = {np.mean(np.abs(diff)):.2e}")
print(f"  Max  |diff| = {np.max(np.abs(diff)):.2e}")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(result_A, cmap="gray")
axes[0].set_title(f"Single Gaussian (t={t_big})")
axes[1].imshow(result_B, cmap="gray")
axes[1].set_title(f"10× Gaussian (t={t_small})")
im_d = axes[2].imshow(diff, cmap="bwr")
axes[2].set_title("Difference")
plt.colorbar(im_d, ax=axes[2])
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()


# =============================================================================
# TASK 5: Verify semi-group for Gaussian derivative (eq 1.9)
#
# dg/dx(t1+t2) * I  =  dg/dx(t1) * [ g(t2) * I ]
#
# Test: derivative at t=20 vs. smooth with t=10 then derivative at t=10
# =============================================================================

print("=" * 60)
print("TASK 5: Verify Semi-Group (Derivative)")
print("=" * 60)

t_total = 20
t_half = 10

# --- Method A: single large Gaussian derivative ---
_, gd_big = create_gaussian_derivative_kernel(np.sqrt(t_total))
result_A = scipy.ndimage.convolve(
    im, gd_big.reshape(1, -1), mode="nearest"
)

# --- Method B: smooth with g(t=10), then derivative with dg(t=10) ---
_, g_half = create_gaussian_kernel(np.sqrt(t_half))
_, gd_half = create_gaussian_derivative_kernel(np.sqrt(t_half))

im_smooth = scipy.ndimage.convolve(
    im, g_half.reshape(1, -1), mode="nearest"
)
result_B = scipy.ndimage.convolve(
    im_smooth, gd_half.reshape(1, -1), mode="nearest"
)

# Compare
diff = result_A - result_B
print(f"  Method A: Gaussian derivative at t={t_total}")
print(f"  Method B: Gaussian(t={t_half}) + derivative(t={t_half})")
print(f"  Mean |diff| = {np.mean(np.abs(diff)):.2e}")
print(f"  Max  |diff| = {np.max(np.abs(diff)):.2e}")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(result_A, cmap="RdBu_r")
axes[0].set_title(f"Gauss deriv (t={t_total})")
axes[1].imshow(result_B, cmap="RdBu_r")
axes[1].set_title(f"Smooth(t={t_half}) + deriv(t={t_half})")
im_d = axes[2].imshow(diff, cmap="bwr")
axes[2].set_title("Difference")
plt.colorbar(im_d, ax=axes[2])
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()


# =============================================================================
# BONUS: Compare with scipy.ndimage.gaussian_filter
# =============================================================================

print("=" * 60)
print("BONUS: Comparison with scipy.ndimage.gaussian_filter")
print("=" * 60)

scipy_result = scipy.ndimage.gaussian_filter(
    im, sigma=sigma, truncate=4
)
our_result = scipy.ndimage.convolve(im, g_2d, mode="nearest")

diff = scipy_result - our_result
print(f"  Mean |diff| = {np.mean(np.abs(diff)):.2e}")
print(f"  Max  |diff| = {np.max(np.abs(diff)):.2e}")
print()
print("Exercise 1.1.1 complete!")





# =============================================================================
# QUIZ: Reveal hidden number in noisy_number_2023.png
#
# The image contains a number hidden under high-frequency noise.
# A large Gaussian blur removes the noise (high frequency) while
# preserving the number (low frequency), making it visible.
# =============================================================================

print("=" * 60)
print("QUIZ: Appearing Number in noisy_number_2023.png")
print("=" * 60)

# Load the noisy image
im_noisy = skimage.io.imread("week1/week1_data/noisy_number_2023.png")
im_noisy = im_noisy.astype(float)
print(f"Image shape: {im_noisy.shape}, dtype: {im_noisy.dtype}")

# Try several sigma values — the number becomes visible at large sigma
sigmas = [1, 3, 5, 10, 20, 30]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, s in zip(axes.flat, sigmas):
    # Build our own 1D Gaussian kernel (reusing our helper function)
    _, g_quiz = create_gaussian_kernel(s)

    # Apply separable 2D smoothing (horizontal then vertical)
    smoothed = scipy.ndimage.convolve(
        im_noisy, g_quiz.reshape(1, -1), mode="nearest"
    )
    smoothed = scipy.ndimage.convolve(
        smoothed, g_quiz.reshape(-1, 1), mode="nearest"
    )

    ax.imshow(smoothed, cmap="gray")
    ax.set_title(f"σ = {s}")
    ax.axis("off")

plt.suptitle(
    "Quiz: Smoothing noisy_number_2023.png to reveal hidden number",
    fontsize=14,
)
plt.tight_layout()
plt.show()