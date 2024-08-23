import numpy as np
import matplotlib.pyplot as plt

# Example cluster centroids (RGB format) from your clustering algorithm
cluster_centroids = np.array([
    [128, 128, 128],  # Example centroid 1
    [60, 179, 113],   # Example centroid 2
    [135, 206, 250],  # Example centroid 3
    [255, 0, 0],      # Example centroid 4
    [0, 255, 0],      # Example centroid 5
    [0, 0, 255],      # Example centroid 6
    [255, 255, 0]     # Example centroid 7
])

# Define known representative colors for your output classes
output_classes_colors = np.array([
    [135, 206, 250],  # Sky
    [34, 139, 34],    # Vegetation
    [70, 130, 180],   # Water
    [169, 169, 169],  # Buildings
    [128, 128, 0],    # Road
    [255, 255, 255],  # Snow/Ice
    [255, 165, 0]     # Desert/Sand
])

# Compute Euclidean distances between each centroid and each output class color
distances = np.linalg.norm(cluster_centroids[:, np.newaxis] - output_classes_colors, axis=2)

# Assign each cluster to the nearest output class based on the minimum distance
assigned_classes = np.argmin(distances, axis=1)

# Visualization: Plotting the centroids and corresponding output classes
fig, ax = plt.subplots()

# Plot cluster centroids
for i, color in enumerate(cluster_centroids):
    ax.scatter(i, 1, color=color/255.0, s=200, label=f'Cluster {i+1}')
    ax.text(i, 1.1, f'Centroid {i+1}', ha='center', va='center')

# Plot output class colors
for i, color in enumerate(output_classes_colors):
    ax.scatter(i, 0, color=color/255.0, s=200, label=f'Class {i+1}')
    ax.text(i, -0.1, f'Class {i+1}', ha='center', va='center')

# Connect the cluster centroids with their assigned output class
for i, (cluster, assigned_class) in enumerate(zip(cluster_centroids, assigned_classes)):
    ax.plot([i, assigned_class], [1, 0], 'k--')

# Formatting the plot
ax.set_ylim(-0.2, 1.2)
ax.set_xlim(-0.5, len(cluster_centroids)-0.5)
ax.set_yticks([])
ax.set_xticks(range(len(cluster_centroids)))
ax.set_xlabel('Cluster Centroids and Output Classes')
ax.set_title('Cluster Centroids vs. Output Classes')

plt.show()
