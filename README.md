# k-means-calib
A prototype implementation of kmeans clustering on the calibration of a vss soccer vision system

## Dependencies
- `numpy` : for numerical operations
- `opencv-python` : for image processing
- `matplotlib` and...
- `plotly` : for visualizations

> just `pip install _` everything before running the code

## Features
- [X] Visualization
- [X] Basic K-Means clustering
- [X] Filtering the floor background based on field positions
- [X] Tackle the field background problem (black and white)
    - idea: remove pixels within a certain distance from the black-white line
- [ ] ~Clustering with lots of samples (removing rare pixels)~
    - abandoned due to badness of spinnaker python api
- [ ] Do some morphological operations on the unique_pixels (smoothening the blobs)
- [ ] Calibration of VSS soccer vision system (implementation in c++)
    - not planning to be add it here. Maybe just the kmeansDoer class
- [ ] Optimization?
