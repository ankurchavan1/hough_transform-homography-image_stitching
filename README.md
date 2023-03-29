# Project Description
This project consists of two problems:

1. Detecting corners of a paper in a video by implementing hough transform from scratch, computing homography between real
world points and pixel coordinates of the corners, and computing the rotation and translation between the camera and a coordinate frame whose
origin is located on any one corner of the sheet of paper.
2. Extracting features from each frame, matching the features between each consecutive image and visualizing them, 
Computing the homographies between the pairs of images, and combining these frames together using the computed homographies to create a panoramic image.

# The following libraries are used:

1. numpy
2. pandas
3. matplotlib
4. seaborn
5. mpl_toolkits.mplot3d
6. cv2

The project includes a report in PDF format and a separate code file. The report provides a detailed explanation of the approach, results, and problems encountered for both problems. 

The code file contains the Python code used to perform the analysis.

# Instructions
To run the code, follow these instructions:

Install Anaconda and create a new environment
Open Jupyter Notebook and navigate to the folder containing the code file
Open the code file in Jupyter Notebook
Place the following files in the same folder as the code file:

1. project2.avi
2. image_1.jpg
3. image_2.jpg
4. image_3.jpg
5. image_4.jpg

Run the code file in Jupyter Notebook

For Problem 1, if you want to see the results of canny edge detection, you can uncomment the line " #cv2.imshow('Canny Edge Detection', edges)" to see the detected edges.

# Results

The implementation of hough transform was successful in detecting the corners of the paper in the video, and the homography was computed accurately. The rotation and translation between the camera and a coordinate frame whose origin is located on any one corner of the sheet of paper were also computed accurately.

For Problem 2, features were successfully extracted from each frame, and matching of features between consecutive images was implemented. Homographies were accurately computed between the pairs of images, and the frames were combined to create a panoramic image

# Conclusion

The project successfully implemented the hough transform for corner detection in a video and computed the homography and rotation and translation between the camera and a coordinate frame whose origin is located on any one corner of the sheet of paper. Additionally, features were extracted from each frame in Problem 2, and matching of features between consecutive images was implemented to create a panoramic image.
