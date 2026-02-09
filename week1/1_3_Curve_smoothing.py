import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant


# =============================================================================
# EXERCISE 1.1.3: Curve Smoothing (Snakes)
#
# CONCEPT:
# A "snake" is a closed curve defined by N ordered 2D points.
# We store them in an N×2 matrix X, where column 0 = x coords,
# column 1 = y coords.
#
# Smoothing moves each point toward the average of its two
# neighbors. Because the curve is CLOSED, the first point's
# left neighbor is the last point, and vice versa.
#
# Three approaches:
#   1. Explicit:  X_new = (I + λL) X          — simple but unstable
#   2. Implicit:  X_new = (I - λL)^{-1} X     — stable, one step
#   3. Extended:  X_new = (I - αA - βB)^{-1} X — controls length & curvature
#
# L is the "Laplacian matrix" — a circulant matrix with the
# pattern [−2, 1, 0, ..., 0, 1] in each row. Multiplying L by X
# computes, for each point, the vector: (left_neighbor − 2·self + right_neighbor),
# which is a discrete approximation of the second derivative.
# Adding this to the point moves it toward its neighbors' average.
# =============================================================================


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_laplacian(N):
    """
    Build the N×N Laplacian matrix L for a closed curve with N points.

    L is circulant with first column: [-2, 1, 0, ..., 0, 1]

    For N=6 this looks like:
        [-2  1  0  0  0  1]
        [ 1 -2  1  0  0  0]
        [ 0  1 -2  1  0  0]
        [ 0  0  1 -2  1  0]
        [ 0  0  0  1 -2  1]
        [ 1  0  0  0  1 -2]

    The 1 in the top-right and bottom-left corners is what makes
    the curve "closed" (wraps around).

    Parameters
    ----------
    N : int
        Number of points on the curve.

    Returns
    -------
    L : ndarray (N×N)
        Laplacian matrix.
    """
    col = np.zeros(N)
    col[0] = -2  # center (the point itself)
    col[1] = 1   # right neighbor
    col[-1] = 1  # left neighbor (wraps around for closed curve)
    return circulant(col)


def make_rigidity_matrix(N):
    """
    Build the N×N rigidity matrix B for a closed curve with N points.

    B is circulant with first column: [-6, 4, -1, 0, ..., 0, -1, 4]

    This corresponds to the 4th-order finite difference kernel
    [-1, 4, -6, 4, -1] which approximates the 4th derivative.
    Minimizing this term penalizes high curvature (sharp bends).

    Parameters
    ----------
    N : int
        Number of points on the curve.

    Returns
    -------
    B : ndarray (N×N)
        Rigidity matrix.
    """
    col = np.zeros(N)
    col[0] = -6   # center
    col[1] = 4    # 1 step right
    col[2] = -1   # 2 steps right
    col[-1] = 4   # 1 step left  (wraps around)
    col[-2] = -1  # 2 steps left (wraps around)
    return circulant(col)


def make_smoothing_matrix(N, alpha, beta):
    """
    Build the smoothing matrix (I - αA - βB)^{-1} for implicit
    curve smoothing with elasticity and rigidity terms.

    Parameters
    ----------
    N : int
        Number of curve points.
    alpha : float
        Elasticity weight (penalizes curve length).
    beta : float
        Rigidity weight (penalizes curvature).

    Returns
    -------
    S : ndarray (N×N)
        Smoothing matrix. Apply as: X_new = S @ X
    """
    I = np.eye(N)
    A = make_laplacian(N)
    B = make_rigidity_matrix(N)
    return np.linalg.inv(I - alpha * A - beta * B)


# =============================================================================
# LOAD CURVE DATA
# =============================================================================

# Adjust paths to match your folder structure
dino_noisy = np.loadtxt("week1/week1_data/curves/dino_noisy.txt")
hand_noisy = np.loadtxt("week1/week1_data/curves/hand_noisy.txt")

# Optional: load clean curves for visual comparison
try:
    dino_clean = np.loadtxt("week1/week1_data/curves/dino.txt")
    hand_clean = np.loadtxt("week1/week1_data/curves/hand.txt")
    has_clean = True
except FileNotFoundError:
    has_clean = False
    print("Clean curve files not found — skipping clean overlay.")

print(f"Dino noisy: {dino_noisy.shape[0]} points")
print(f"Hand noisy: {hand_noisy.shape[0]} points")
print()


# =============================================================================
# HELPER: Plot a closed curve
# =============================================================================

def plot_curve(ax, X, **kwargs):
    """
    Plot a closed curve. Appends the first point at the end
    so the curve visually closes.
    """
    # Append first point to close the loop visually
    closed = np.vstack([X, X[0]])
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


# =============================================================================
# TASK 1: Explicit smoothing  X_new = (I + λL) X
#
# With λ = 0.5, each point moves exactly to the average of its
# two neighbors, because:
#   x_new = x + 0.5 * (x_left - 2*x + x_right)
#         = x + 0.5*x_left - x + 0.5*x_right
#         = 0.5 * (x_left + x_right)
# =============================================================================

print("=" * 60)
print("TASK 1: Explicit Smoothing")
print("=" * 60)

X = dino_noisy.copy()
N = X.shape[0]

# Build the Laplacian matrix
L = make_laplacian(N)
I_mat = np.eye(N)  # identity matrix (named I_mat to avoid shadowing)

# --- Single step with λ = 0.5 ---
lam = 0.5
X_one_step = (I_mat + lam * L) @ X

# Verify: each point should be the average of its neighbors
# Check a few points manually
print(f"λ = {lam}, single step verification:")
for idx in [0, 1, N - 1]:
    left = (idx - 1) % N   # wrap around for closed curve
    right = (idx + 1) % N
    neighbor_avg = 0.5 * (X[left] + X[right])
    print(f"  Point {idx}: result = {X_one_step[idx]}, "
          f"neighbor avg = {neighbor_avg}, "
          f"match = {np.allclose(X_one_step[idx], neighbor_avg)}")
print()

# --- Iterative smoothing with small λ ---
# Try different λ values and iteration counts

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

configs = [
    (0.1, 5, "λ=0.1, 5 iters"),
    (0.1, 50, "λ=0.1, 50 iters"),
    (0.1, 200, "λ=0.1, 200 iters"),
    (0.5, 5, "λ=0.5, 5 iters"),
    (0.5, 50, "λ=0.5, 50 iters"),
    (0.5, 200, "λ=0.5, 200 iters"),
]

for ax, (lam, n_iter, title) in zip(axes.flat, configs):
    X_iter = dino_noisy.copy()
    smoothing_matrix = I_mat + lam * L

    for _ in range(n_iter):
        X_iter = smoothing_matrix @ X_iter

    # Plot original noisy curve
    plot_curve(ax, dino_noisy, color="red", alpha=0.3,
               linewidth=1, label="Noisy")
    # Plot smoothed curve
    plot_curve(ax, X_iter, color="blue", linewidth=2,
               label="Smoothed")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

plt.suptitle("Task 1: Explicit Smoothing — Effect of λ and Iterations",
             fontsize=14)
plt.tight_layout()
plt.show()

# --- Demonstrate instability with large λ ---
print("Demonstrating instability with large λ:")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
unstable_lambdas = [0.3, 0.5, 0.8]

for ax, lam in zip(axes, unstable_lambdas):
    X_iter = dino_noisy.copy()
    smoothing_matrix = I_mat + lam * L

    for _ in range(100):
        X_iter = smoothing_matrix @ X_iter

    plot_curve(ax, dino_noisy, color="red", alpha=0.3,
               linewidth=1, label="Noisy")
    plot_curve(ax, X_iter, color="blue", linewidth=2,
               label="Smoothed")
    ax.set_title(f"λ = {lam}, 100 iterations")
    ax.set_aspect("equal")
    ax.legend()

plt.suptitle("Task 1: Larger λ Can Cause Oscillations or Shrinkage",
             fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# TASK 2: Implicit smoothing  X_new = (I - λL)^{-1} X
#
# The key insight: instead of computing displacement based on the
# CURRENT curve (explicit), we set up an equation where the
# displacement is evaluated on the NEW curve (implicit / backward
# Euler). This makes it unconditionally stable — no oscillations
# regardless of how large λ is.
#
# The trade-off: we need to invert a matrix, but for a given N
# and λ this only needs to be done once.
# =============================================================================

print("=" * 60)
print("TASK 2: Implicit Smoothing")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Test with various λ values — even very large ones work!
lambdas = [0.5, 2, 10, 50, 200, 1000]

for ax, lam in zip(axes.flat, lambdas):
    # Precompute the smoothing matrix (only depends on N and λ)
    smooth_mat = np.linalg.inv(I_mat - lam * L)

    # Apply in a single step — no iterations needed!
    X_smooth = smooth_mat @ dino_noisy

    plot_curve(ax, dino_noisy, color="red", alpha=0.3,
               linewidth=1, label="Noisy")
    plot_curve(ax, X_smooth, color="blue", linewidth=2,
               label="Smoothed")
    ax.set_title(f"λ = {lam} (single step)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

plt.suptitle("Task 2: Implicit Smoothing — Stable for Any λ\n"
             "(Note: curve shrinks as λ increases)",
             fontsize=14)
plt.tight_layout()
plt.show()

print("Observation: Implicit smoothing is stable for any λ,")
print("but the curve still SHRINKS because the kernel minimizes length.")
print("→ This motivates the extended kernel in Task 3.")
print()


# =============================================================================
# TASK 3: Implicit smoothing with extended kernel
#         X_new = (I - αA - βB)^{-1} X
#
# Two terms control the smoothing:
#   α (elasticity) — penalizes curve LENGTH
#                    kernel: [0, 1, -2, 1, 0]  (same as Laplacian L)
#   β (rigidity)   — penalizes curve CURVATURE
#                    kernel: [-1, 4, -6, 4, -1]
#
# A is identical to L (Laplacian).
# B encodes the rigidity kernel.
#
# By choosing large β and small α, we can smooth the curve
# (remove noise / reduce curvature) without excessive shrinkage.
# =============================================================================

print("=" * 60)
print("TASK 3: Extended Kernel (Elasticity + Rigidity)")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Test different combinations of α and β
configs = [
    (1.0, 0.0, "α=1.0, β=0.0\n(elasticity only = Task 2)"),
    (0.0, 1.0, "α=0.0, β=1.0\n(rigidity only)"),
    (0.1, 1.0, "α=0.1, β=1.0\n(small α, large β)"),
    (1.0, 1.0, "α=1.0, β=1.0\n(equal weights)"),
    (0.01, 10.0, "α=0.01, β=10.0\n(very rigid)"),
    (1.0, 0.1, "α=1.0, β=0.1\n(large α, small β)"),
]

for ax, (alpha, beta, title) in zip(axes.flat, configs):
    # Build smoothing matrix using the helper function
    smooth_mat = make_smoothing_matrix(N, alpha, beta)

    # Single-step smoothing
    X_smooth = smooth_mat @ dino_noisy

    plot_curve(ax, dino_noisy, color="red", alpha=0.3,
               linewidth=1, label="Noisy")
    plot_curve(ax, X_smooth, color="blue", linewidth=2,
               label="Smoothed")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

plt.suptitle("Task 3: Extended Kernel — α (length) vs β (curvature)",
             fontsize=14)
plt.tight_layout()
plt.show()

print("Observations:")
print("  • β only (rigidity):   smooths bends, preserves overall shape")
print("  • α only (elasticity): shrinks the curve (minimizes length)")
print("  • Large β, small α:    best for denoising without shrinkage")
print()


# =============================================================================
# TASK 4: Reusable smoothing matrix function
#
# The exercise asks us to implement a function that, given N, α, β,
# returns (I - αA - βB)^{-1}. We already did this above as
# make_smoothing_matrix(). Let's verify it works on both curves.
# =============================================================================

print("=" * 60)
print("TASK 4: Reusable Smoothing Function on Both Curves")
print("=" * 60)

# Good default parameters for denoising
alpha = 0.05
beta = 5.0

curves = {
    "Dino": dino_noisy,
    "Hand": hand_noisy,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (name, X_noisy) in zip(axes, curves.items()):
    N_curve = X_noisy.shape[0]

    # Build smoothing matrix for this curve size
    smooth_mat = make_smoothing_matrix(N_curve, alpha, beta)

    # Apply smoothing
    X_smooth = smooth_mat @ X_noisy

    # Plot
    plot_curve(ax, X_noisy, color="red", alpha=0.4,
               linewidth=1, label="Noisy")
    plot_curve(ax, X_smooth, color="blue", linewidth=2,
               label=f"Smoothed (α={alpha}, β={beta})")

    # If clean curve available, show for reference
    if has_clean:
        X_clean = dino_clean if name == "Dino" else hand_clean
        plot_curve(ax, X_clean, color="green", linewidth=1,
                   linestyle="--", alpha=0.6, label="Clean (reference)")

    ax.set_title(f"{name} Curve")
    ax.set_aspect("equal")
    ax.legend()

plt.suptitle("Task 4: Smoothing Function Applied to Both Curves",
             fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# BONUS: Animate the explicit smoothing to see the curve evolve
#
# This creates a simple step-by-step visualization showing how
# explicit smoothing gradually denoises (and shrinks) the curve.
# =============================================================================

print("=" * 60)
print("BONUS: Step-by-Step Explicit Smoothing Visualization")
print("=" * 60)

lam = 0.3
n_steps = 100
snapshot_interval = 10  # show every N-th step

X_iter = dino_noisy.copy()
N_dino = X_iter.shape[0]
L_dino = make_laplacian(N_dino)
I_dino = np.eye(N_dino)
step_matrix = I_dino + lam * L_dino

# Collect snapshots
snapshots = [(0, dino_noisy.copy())]
for step in range(1, n_steps + 1):
    X_iter = step_matrix @ X_iter
    if step % snapshot_interval == 0:
        snapshots.append((step, X_iter.copy()))

# Plot snapshots
n_snaps = len(snapshots)
cols = min(n_snaps, 5)
rows = (n_snaps + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = np.array(axes).flatten()  # handle single-row case

for i, (step, X_snap) in enumerate(snapshots):
    ax = axes[i]
    plot_curve(ax, dino_noisy, color="red", alpha=0.2,
               linewidth=1, label="Original")
    plot_curve(ax, X_snap, color="blue", linewidth=2,
               label=f"Step {step}")
    ax.set_title(f"Step {step}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

# Hide unused subplots
for i in range(n_snaps, len(axes)):
    axes[i].set_visible(False)

plt.suptitle(f"Explicit Smoothing Evolution (λ={lam})", fontsize=14)
plt.tight_layout()
plt.show()

print()
print("Exercise 1.1.3 complete!")









# =============================================================================
# QUIZ: Curve length after explicit smoothing of dino_noisy.txt
#
# Apply Equation 1.10:  X_new = (I + λL) X  with λ = 0.19
# Then compute the total arc length of the smoothed closed curve:
#   length = sum of Euclidean distances between consecutive points
#            (including last point → first point to close the curve)
# =============================================================================

print("=" * 60)
print("QUIZ: Curve Length after Explicit Smoothing (Eq. 1.10)")
print("=" * 60)


def curve_length(X):
    """
    Compute the total arc length of a closed curve.

    The length is the sum of Euclidean distances between
    consecutive points, including the closing segment
    from the last point back to the first.

    Parameters
    ----------
    X : ndarray (N×2)
        Curve points (x, y coordinates).

    Returns
    -------
    length : float
        Total arc length of the closed curve.
    """
    # np.roll shifts all points by 1 position, so:
    #   X_rolled[0] = X[1], X_rolled[1] = X[2], ..., X_rolled[-1] = X[0]
    # This pairs each point with its next neighbor (wrapping around)
    X_rolled = np.roll(X, -1, axis=0)

    # Euclidean distance between each point and the next
    segment_lengths = np.sqrt(np.sum((X_rolled - X) ** 2, axis=1))

    return np.sum(segment_lengths)


# Load the noisy dino curve
X_quiz = np.loadtxt("week1/week1_data/curves/dino_noisy.txt")
N_quiz = X_quiz.shape[0]

print(f"Number of points: {N_quiz}")
print(f"Original curve length: {curve_length(X_quiz):.1f}")

# Build matrices
I_quiz = np.eye(N_quiz)
L_quiz = make_laplacian(N_quiz)

# Apply single step of explicit smoothing (Equation 1.10)
lam_quiz = 0.19
X_smoothed = (I_quiz + lam_quiz * L_quiz) @ X_quiz

# Compute arc length of smoothed curve
length = curve_length(X_smoothed)

print(f"λ = {lam_quiz}")
print(f"Smoothed curve length = {length:.1f}")
print()

# Check against provided options
options = {
    "a": 27.3,  "b": 31.3,  "c": 43.1,  "d": 56.6,
    "e": 62.6,  "f": 76.4,  "g": 79.5,  "h": 80.0,
    "i": 86.2,  "j": 95.5,  "k": 143.6, "l": 152.8,
    "m": 159.0, "n": 160.0, "o": 191.0, "p": 287.2,
    "q": 31249.0, "r": 62498.0,
}

for key, val in options.items():
    if abs(val - length) < 0.5:  # tolerance for rounding
        print(f"  ✓ Answer: ({key}) {val}")
        break
else:
    closest = min(options.items(), key=lambda kv: abs(kv[1] - length))
    print(f"  No exact match. Closest: ({closest[0]}) {closest[1]}")
    print(f"  Our value: {length:.1f}, difference: {abs(closest[1] - length):.1f}")