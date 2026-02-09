import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# =============================================================================
# EXERCISE 1.1.2: Computing Length of Segmentation Boundary
#
# CONCEPT:
# A segmentation image S(x,y) assigns a label (integer) to each pixel.
# The "boundary" is where two neighboring pixels have different labels.
# We use 4-connectivity: each pixel has 4 neighbors (up, down, left, right).
#
# Boundary length L(S) counts all pairs of adjacent pixels with
# different labels. Each pair is counted exactly once.
#
# Example (3x3 image with 2 labels):
#
#     1 1 2        Horizontal boundaries (|):  1 1|2    → 1
#     1 2 2                                    1|2 2    → 1
#     1 1 1                                    1 1 1    → 0
#                                                   total: 2
#
#                  Vertical boundaries (—):    1 1 2
#                                              — —
#                                              1 2 2    → 1 (middle column)
#                                              —
#                                              1 1 1    → 1 (middle column)
#                                                   total: 2
#
#     Total boundary length = 2 + 2 = 4
# =============================================================================


# =============================================================================
# HELPER FUNCTION: Boundary length (vectorized)
# =============================================================================

def boundary_length(S):
    """
    Compute the length of the segmentation boundary using
    4-connectivity. The length equals the number of adjacent
    pixel pairs that have different labels.

    Uses numpy vectorization (no loops) for efficiency.

    Parameters
    ----------
    S : ndarray (2D)
        Segmentation image where each pixel has an integer label.

    Returns
    -------
    length : int
        Total boundary length (number of label transitions).
    """
    # --- Vertical edges (compare each row with the row below it) ---
    #
    # S[:-1, :] = all rows except the last
    # S[1:,  :] = all rows except the first
    #
    # These two arrays are the same size. Comparing element-wise
    # checks whether pixel (r, c) differs from pixel (r+1, c).
    #
    #   S[:-1, :]     S[1:, :]
    #   row 0         row 1       ← compare these
    #   row 1         row 2       ← compare these
    #   ...           ...
    #   row N-2       row N-1     ← compare these
    #
    vertical = np.sum(S[:-1, :] != S[1:, :])

    # --- Horizontal edges (compare each column with the column to its right) ---
    #
    # S[:, :-1] = all columns except the last
    # S[:, 1:]  = all columns except the first
    #
    # Checks whether pixel (r, c) differs from pixel (r, c+1).
    #
    horizontal = np.sum(S[:, :-1] != S[:, 1:])

    # Total boundary length = vertical + horizontal transitions
    return vertical + horizontal


# =============================================================================
# LOAD SEGMENTATION IMAGES
# =============================================================================

# Adjust these paths to match your folder structure
filenames = [
    "week1/week1_data/fuel_cells/fuel_cell_1.tif",
    "week1/week1_data/fuel_cells/fuel_cell_2.tif",
    "week1/week1_data/fuel_cells/fuel_cell_3.tif",
]

# Load all images into a list
images = []
for f in filenames:
    img = skimage.io.imread(f)
    images.append(img)
    print(f"Loaded {f}: shape={img.shape}, dtype={img.dtype}, "
          f"unique labels={np.unique(img)}")

print()


# =============================================================================
# TASK 1 & 2: Compute boundary length for all images (vectorized)
# =============================================================================

print("=" * 60)
print("TASK 1 & 2: Boundary Length (vectorized, no loops)")
print("=" * 60)

for i, (img, fname) in enumerate(zip(images, filenames)):
    length = boundary_length(img)
    n_labels = len(np.unique(img))

    # Also compute total pixels and boundary as percentage
    total_pixels = img.shape[0] * img.shape[1]
    boundary_pct = 100 * length / total_pixels

    print(f"  Image {i+1} ({fname.split('/')[-1]}):")
    print(f"    Number of labels : {n_labels}")
    print(f"    Boundary length  : {length}")
    print(f"    Image size       : {img.shape[0]} × {img.shape[1]} "
          f"= {total_pixels} pixels")
    print(f"    Boundary / total : {boundary_pct:.2f}%")
    print()


# =============================================================================
# VISUALIZE the segmentation images and their boundaries
# =============================================================================

fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))

# Handle case where there's only one image (axes wouldn't be a list)
if len(images) == 1:
    axes = [axes]

for i, (img, fname) in enumerate(zip(images, filenames)):
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(
        f"{fname.split('/')[-1]}\nL = {boundary_length(img)}"
    )
    axes[i].axis("off")

plt.suptitle("Segmentation Images and Boundary Lengths", fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# BONUS: Visualize where the boundaries actually are
#
# We can create a boundary image by marking pixels that have at
# least one neighbor with a different label.
# =============================================================================

def boundary_image(S):
    """
    Create a binary image showing boundary pixels.
    A pixel is on the boundary if any of its 4-neighbors has a
    different label.

    Parameters
    ----------
    S : ndarray (2D)
        Segmentation image.

    Returns
    -------
    boundary : ndarray (2D, bool)
        True where pixel is on a boundary.
    """
    # Start with all False (no boundary)
    boundary = np.zeros_like(S, dtype=bool)

    # Check each of the 4 directions and mark boundaries
    # Down:  compare row r with row r+1
    boundary[:-1, :] |= (S[:-1, :] != S[1:, :])
    # Up:    the same comparison, but mark the bottom pixel too
    boundary[1:, :]  |= (S[:-1, :] != S[1:, :])
    # Right: compare col c with col c+1
    boundary[:, :-1] |= (S[:, :-1] != S[:, 1:])
    # Left:  same comparison, mark the right pixel too
    boundary[:, 1:]  |= (S[:, :-1] != S[:, 1:])

    return boundary


fig, axes = plt.subplots(2, len(images), figsize=(5 * len(images), 10))

for i, (img, fname) in enumerate(zip(images, filenames)):
    # Top row: original segmentation
    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].set_title(f"{fname.split('/')[-1]}")
    axes[0, i].axis("off")

    # Bottom row: boundary pixels highlighted
    bnd = boundary_image(img)
    axes[1, i].imshow(img, cmap="gray")
    # Overlay boundary in red using a masked array
    overlay = np.zeros((*img.shape, 4))  # RGBA
    overlay[bnd] = [1, 0, 0, 0.6]       # red, semi-transparent
    axes[1, i].imshow(overlay)
    axes[1, i].set_title(f"Boundaries (L = {boundary_length(img)})")
    axes[1, i].axis("off")

plt.suptitle("Segmentations with Boundary Overlay", fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# UNDERSTANDING CHECK: Compare vectorized vs. naive loop approach
#
# This section exists purely to demonstrate that the vectorized
# version gives the same result as the obvious (but slow) loop.
# You would never use the loop version in practice.
# =============================================================================

def boundary_length_loop(S):
    """
    Compute boundary length using explicit loops.
    Slow but easy to understand — useful for verification only.
    """
    rows, cols = S.shape
    length = 0

    for r in range(rows):
        for c in range(cols):
            # Check neighbor below (avoid going out of bounds)
            if r + 1 < rows and S[r, c] != S[r + 1, c]:
                length += 1
            # Check neighbor to the right
            if c + 1 < cols and S[r, c] != S[r, c + 1]:
                length += 1

    return length


print("=" * 60)
print("VERIFICATION: Vectorized vs. Loop Implementation")
print("=" * 60)

for i, img in enumerate(images):
    l_vec = boundary_length(img)
    l_loop = boundary_length_loop(img)
    match = "✓" if l_vec == l_loop else "✗ MISMATCH"
    print(f"  Image {i+1}: vectorized={l_vec}, loop={l_loop}  {match}")

print()
print("Exercise 1.1.2 complete!")








# =============================================================================
# QUIZ: Boundary length of fuel_cell_2.tif
#
# We simply call our boundary_length() function on the image
# and match the result to one of the given options.
# =============================================================================

print("=" * 60)
print("QUIZ: Boundary Length of fuel_cell_2.tif")
print("=" * 60)

# Load the specific image
im_quiz = skimage.io.imread("week1/week1_data/fuel_cells/fuel_cell_2.tif")

# Print image info for sanity check
print(f"Image shape : {im_quiz.shape}")
print(f"Image dtype : {im_quiz.dtype}")
print(f"Unique labels: {np.unique(im_quiz)}")
print(f"Num labels   : {len(np.unique(im_quiz))}")

# Compute boundary length using our function
L = boundary_length(im_quiz)

print(f"\nBoundary length of fuel_cell_2.tif = {L}")
print()

# Check against the provided options
options = {
    "a": 277,
    "b": 554,
    "c": 5524,
    "d": 8084,
    "e": 10554,
    "f": 10801,
    "g": 11048,
    "h": 12328,
    "i": 16168,
    "j": 21602,
    "k": 24656,
}

for key, val in options.items():
    if val == L:
        print(f"  ✓ Answer: ({key}) {val}")
        break
else:
    # If no exact match, show closest option
    closest = min(options.items(), key=lambda kv: abs(kv[1] - L))
    print(f"  No exact match. Closest: ({closest[0]}) {closest[1]}")
    print(f"  Difference: {abs(closest[1] - L)}")