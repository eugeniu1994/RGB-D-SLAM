# RGB-D SLAM
This is an example of indoor RGB-D SLAM
1. Detect and Track image feature
2. Perform Ransac PnP algorithm to estimate the transformation between the 3D point cloud on the current frame and their 2D pixel correspondences in the previous frame.
3. Build a pose graph based on estimated transformations
4. Update the current pose based on estimated R and t.
5. Build the sparse map


<img src="rgbd.gif" />

## To do:
* add Bundle adjustment
* add loop closure detection with V-BoW

