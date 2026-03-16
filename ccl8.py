import cv2
import numpy as np

# Load image and threshold it to get a binary image (0 and 255)
# Using your previous thresholding logic
img = cv2.imread("F:\Languages\DIP\est1\ig02.tif", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
rows, cols = binary.shape

# Initialize labels matrix with zeros
labels = np.zeros((rows, cols), dtype=np.int32)
next_label = 1
# Equivalency dictionary to track which labels belong to the same object
parent = {}

def find(i):
    """Find the root label for an equivalency."""
    while parent[i] != i:
        i = parent[i]
    return i

def union(i, j):
    """Merge two labels into one equivalency group."""
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

# --- PASS 1: Assign Temporary Labels ---
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        if binary[i][j] == 255:  # Foreground pixel
            # Check 8-neighbors (Past neighbors: W, NW, N, NE)
            neighbor_labels = [
                labels[i][j-1],   # West
                labels[i-1][j-1], # North-West
                labels[i-1][j],   # North
                labels[i-1][j+1]  # North-East
            ]
            
            # Filter out zero labels (background neighbors)
            active_neighbors = [label for label in neighbor_labels if label > 0]
            
            if not active_neighbors:
                # No labeled neighbors -> Start a new component
                labels[i][j] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                # Assign the minimum label found in neighbors
                min_label = min(active_neighbors)
                labels[i][j] = min_label
                # Record equivalency for all other neighbors found
                for neighbor in active_neighbors:
                    union(neighbor, min_label)

# --- PASS 2: Resolve Equivalencies ---
for i in range(rows):
    for j in range(cols):
        if labels[i][j] > 0:
            labels[i][j] = find(labels[i][j])

# Normalization for visualization (scaling labels to 0-255 range)
max_label = labels.max()
out = np.zeros((rows, cols), dtype=np.uint8)
if max_label > 0:
    out = (labels / max_label * 255).astype(np.uint8)

cv2.imshow("Binary Image", binary)
cv2.imshow("Labeled Components", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
