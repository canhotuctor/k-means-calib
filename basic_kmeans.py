import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Load and convert the image
img = cv2.imread("./few_pixels.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of RGB values
pixels = img.reshape(-1, 3)

# Calculate the distance from the black-white diagonal
distances = np.linalg.norm(pixels - np.mean(pixels, axis=1, keepdims=True), axis=1)

# Filter out pixels close to the black-white diagonal
threshold = 25 # Adjust this value as needed
mask = distances > threshold
filtered_pixels = pixels[mask]

# Find unique rows (unique colors) and their counts
unique_pixels, inverse_indices = np.unique(filtered_pixels, axis=0, return_inverse=True)

# Convert unique pixel values to float32 for k-means
unique_pixels = np.float32(unique_pixels)

# Define criteria and apply k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 7  # Number of clusters desired
attempts = 10
aa = time.time()
ret, label, center = cv2.kmeans(unique_pixels, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
bb = time.time()
print("Time to run k-means: ", bb - aa)

# Convert the center (cluster centroids) back to uint8 (RGB)
center = np.uint8(center)

# Replace each pixel in the original image with the centroid of its cluster
segmented_pixels = center[label[inverse_indices]].reshape(-1, 3)

# Reconstruct the full image including the background pixels
result_image = np.zeros_like(pixels)
result_image[mask] = segmented_pixels

# Reshape the segmented pixels back to the original image shape
result_image = result_image.reshape(img.shape)

# Create an image to display the cluster centroids as color boxes
box_width = 100
centroid_image = np.zeros((box_width, K * box_width, 3), dtype=np.uint8)

# Draw the centroids with RGB labels
for i in range(K):
    centroid_image[:, i * box_width:(i + 1) * box_width] = center[i]


# Plot the original, segmented images, and centroids with RGB codes
figure_size = 20

# Create a figure
fig, axs = plt.subplots(1, 2, figsize=(figure_size, figure_size))

# Plot the original image
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[0].set_xticks([])
axs[0].set_yticks([])

# Plot the segmented image
axs[1].imshow(result_image)
axs[1].set_title('Segmented Image when K = %i' % K)
axs[1].set_xticks([])
axs[1].set_yticks([])

fig2, axs2 = plt.subplots(1, 1, figsize=(figure_size, figure_size))

# Plot the centroids
axs2.imshow(centroid_image)
axs2.set_title('Cluster Centroids')
axs2.set_xticks([])
axs2.set_yticks([])

# Add RGB codes below each centroid
for i in range(K):
    axs2.text((i + 0.5) * box_width, box_width + 20, f'RGB: {center[i]}', fontsize=12, ha='center', va='top')
    
plt.tight_layout()
plt.show()