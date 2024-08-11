# k-means-calib
A prototype implementation of kmeans clustering on the calibration of a vss soccer vision system

## Dependencies
- **numpy** : for numerical operations
- **opencv-python** : for image processing
- **matplotlib** and...
- **plotly** : for visualizations

> just pip install everything before running the code

## Features
[X] Visualization

[X] Basic K-Means clustering
[ ] Filtering the floor background based on field positions
[X] Tackle the field background problem (black and white)
    - idea: remove pixels within a certain distance from the black-white line
[ ] Clustering with lots of samples (removing rare pixels)
[ ] Calibration of VSS soccer vision system
[ ] Optimization