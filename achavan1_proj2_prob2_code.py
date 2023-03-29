import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading and resizing the images

img_1 = cv2.imread('image_1.jpg')
scale_percent = 20 # percent of original size
width_1 = int(img_1.shape[1] * scale_percent / 100)
height_1 = int(img_1.shape[0] * scale_percent / 100)
dim_1 = (width_1, height_1) 
# resize image
img1 = cv2.resize(img_1, dim_1, interpolation = cv2.INTER_AREA)

img_2 = cv2.imread('image_2.jpg')
width_2 = int(img_2.shape[1] * scale_percent / 100)
height_2 = int(img_2.shape[0] * scale_percent / 100)
dim_2 = (width_2, height_2) 
# resize image
img2 = cv2.resize(img_2, dim_2, interpolation = cv2.INTER_AREA)

img_3 = cv2.imread('image_3.jpg')
scale_percent = 20 # percent of original size
width_3 = int(img_3.shape[1] * scale_percent / 100)
height_3 = int(img_3.shape[0] * scale_percent / 100)
dim_3 = (width_3, height_3) 
# resize image
img3 = cv2.resize(img_3, dim_3, interpolation = cv2.INTER_AREA)

img_4 = cv2.imread('image_4.jpg')
scale_percent = 20 # percent of original size
width_4 = int(img_4.shape[1] * scale_percent / 100)
height_4 = int(img_4.shape[0] * scale_percent / 100)
dim_4 = (width_4, height_4) 
# resize image
img4 = cv2.resize(img_4, dim_4, interpolation = cv2.INTER_AREA)

# Create SIFT object
sift = cv2.SIFT_create()

# Function to create a panorama:

def stitcher(left_image,right_image, sift):

    gray_img1= cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_img2= cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    

    kp1, desrip1 = sift.detectAndCompute(gray_img1, None)
    kp2, desrip2 = sift.detectAndCompute(gray_img2, None)

    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(desrip1, desrip2, k=2)

    good_matches = []
    for i,j in matches:
        if i.distance < 0.7*j.distance:
            good_matches.append(i)

     
    source_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dest_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    thres = 10
    no_of_iterations = 1000


    best_H_matrix = None
    min_inliers = 0

    for i in range(no_of_iterations):
        random_i = np.random.choice(len(source_pts), 4, replace= False)
        sampl1 = source_pts[random_i]
        sampl2 = dest_pts[random_i]

        A = []
        for i in range(len(sampl2)):
            x, y = sampl2[i][0][0], sampl2[i][0][1]
            u, v = sampl1[i][0][0], sampl1[i][0][1]
            A.append(np.array([
            [x, y, 1, 0, 0, 0, -u*x, -u*y, -u],
            [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
        ]))
            A_mat = np.empty([0, A[0].shape[1]])

        for i in A:
         A_mat = np.append(A_mat, i, axis=0)

   
        eig_val, eig_vec = np.linalg.eig(A_mat.T @ A_mat)

    
        H = eig_vec[:, np.argmin(eig_val)]
   
        H_matrix = H.reshape((3, 3))

        H_ip = np.concatenate(
        (dest_pts, np.ones((len(dest_pts), 1, 1), dtype=np.float32)), axis=2)

        best_H_pts = np.matmul(H_ip, H_matrix.T)

        updated_pts = best_H_pts[:, :, :2] / best_H_pts[:, :, 2:]

       
        diff = np.linalg.norm(source_pts - updated_pts, axis = 2)
        inliners = np.sum(diff<thres)

        if inliners > min_inliers:
            min_inliers = inliners
            best_H_matrix = H_matrix
    
      
    stiched_img = cv2.warpPerspective(right_image, best_H_matrix, ((left_image.shape[1] + right_image.shape[1]), left_image.shape[0]))
    stiched_img[0:left_image.shape[0], 0:left_image.shape[1]] = left_image


    return stiched_img

# Function to find and draw the matches:

def good_matches_ij(img1, img2, sift):

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.2*n.distance:
            good_matches.append(m)
        
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches12', img_matches)

# Visualizing the matches:

good_matches_ij(img1, img2, sift)
cv2.waitKey(0)
cv2.destroyAllWindows()

good_matches_ij(img2, img3, sift)
cv2.waitKey(0)
cv2.destroyAllWindows()

good_matches_ij(img3, img4, sift)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Creating a Panorama:

stitch34 = stitcher(img3,img4, sift)

stitch234 = stitcher(img2, stitch34, sift)

stitch1234 = stitcher(img1, stitch234, sift)

# Displyaing the result:

cv2.imshow('Final Panaroma', stitch1234)
cv2.waitKey(0)
cv2.destroyAllWindows()