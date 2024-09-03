import cv2
import numpy as np
import plotly.graph_objects as go
import time

# Load and convert the image
img = cv2.imread("./normal_field.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Reshape the image to a 2D array of RGB values
pixels = img.reshape(-1, 3)  # Flatten the image, keeping the RGB channels together

# Calculate the distance from the black-white diagonal
distances = np.linalg.norm(pixels - np.mean(pixels, axis=1, keepdims=True), axis=1)

# Filter out pixels close to the black-white diagonal
threshold = 40  # To be adjusted
mask = distances > threshold
pixels = pixels[mask]

# Find unique rows (unique colors) and their counts
bb = time.time()
unique_pixels, pixels_count = np.unique(pixels, axis=0,  return_counts=True)
cc = time.time()
r, g, b = unique_pixels[:, 0], unique_pixels[:, 1], unique_pixels[:, 2]

print("Time to reshape and split image: ", cc - bb)

# Normalize RGB values for Plotly (range 0 to 1)
colors = unique_pixels / 255.0

fig = go.Figure(data=[go.Scatter3d(
    x=r, y=g, z=b,
    mode='markers',
    marker=dict(
        size=np.multiply(np.log(pixels_count), 4),  # Log scale for marker size based on pixel count
        color=['rgb({}, {}, {})'.format(*pixel) for pixel in unique_pixels],
        opacity=0.8,
        line=dict(width=0)
    )
)])

fig.update_layout(
    scene=dict(
        xaxis_title='Red',
        yaxis_title='Green',
        zaxis_title='Blue'
    ),
    title="3D Color Scatter Plot"
)

# Show the plot
fig.show()
