import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the video
cap = cv2.VideoCapture('project2.avi')

# Initializing empty lists
camera_x_translation = []
camera_y_translation = []
camera_z_translation = []

roll_plot = []
pitch_plot = []
yaw_plot = []

frame_count = []

# Loop through each frame
while cap.isOpened():
    # Read the frame
    ret, img_1 = cap.read()
    
    # Check if end of video
    if not ret:
        break
    
    scale_percent = 40 # percent of original size
    width_1 = int(img_1.shape[1] * scale_percent / 100)
    height_1 = int(img_1.shape[0] * scale_percent / 100)
    dim_1 = (width_1, height_1) 
    # resize image
    frame = cv2.resize(img_1, dim_1, interpolation = cv2.INTER_AREA)
    
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5,5),np.uint8)
    morph1_img = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations= 3)
    
    #Convert to graycsale
    img_gray = cv2.cvtColor(morph1_img, cv2.COLOR_BGR2GRAY)    
    
    ret,thresh1 = cv2.threshold(img_gray,205,255,cv2.THRESH_BINARY)
    
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((25,25),np.uint8)
    morph2_img = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations= 3)
 
    #Canny Edge Detection
    edges = cv2.Canny(image=thresh1, threshold1=100, threshold2=200) # Canny Edge Detection

    #edges.shape[0], edges.shape[1] = edges.shape

    rho_resolution = 1
    theta_resolution = 1

    rho_max = int(np.sqrt(edges.shape[0] **2 + edges.shape[1] **2))

    accumulator_space = np.zeros((2*rho_max, (180//theta_resolution)))

    thetas = range(0, 180, theta_resolution)

    for r in range(edges.shape[0]):
      for c in range(edges.shape[1]):
        if edges[r][c] !=0:

          for theta_idx, theta in enumerate(thetas):
            rho = int(c * np.cos(theta * (np.pi/180)) + r * np.sin(theta * (np.pi/180)))
            accumulator_space[rho + rho_max][theta_idx] += 1

    detected_lines = []

    for m in range(0,4):
      highest_votes = 0                 
      max_value_in_accumulator = np.max(accumulator_space)
      idx_of_max_ele = np.argwhere(accumulator_space == max_value_in_accumulator)

      for i in idx_of_max_ele:
          votes = accumulator_space[i[0], i[1]]
          if votes > highest_votes:
              highest_votes = votes
              current_rho = i[0] - rho_max
              current_theta = i[1]
      detected_lines.append((current_rho,current_theta))
      accumulator_space[idx_of_max_ele[:, 0], idx_of_max_ele[:, 1]] = 0

      for j in range(-10, 11):
        for k in range(-10, 11):
          rho_idx = current_rho + j + rho_max
          theta_idx = current_theta+ k
          if 0 <= rho_idx < 2 * rho_max and 0 <= theta_idx < 180:
            accumulator_space[rho_idx, theta_idx] = 0

    for rho, theta in detected_lines:
              
            x1 = int(rho*np.cos(theta* (np.pi/180)) + 1000*(-np.sin(theta*(np.pi/180))))
            y1 = int(rho*np.sin(theta*(np.pi/180)) + 1000*(np.cos(theta* (np.pi/180))))

            x2 = int(rho*np.cos(theta* (np.pi/180)) - 1000*(-np.sin(theta*(np.pi/180))))
            y2 = int(rho*np.sin(theta*(np.pi/180)) - 1000*(np.cos(theta* (np.pi/180))))
          
            cv2.line(morph1_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    corners = []
    
    for i in range(len(detected_lines)):
      for j in range(i+1, len(detected_lines)):
        R_row1, theta1 = detected_lines[i]
        R_row2, theta2 = detected_lines[j]

        A_matrix = np.array([[np.cos(theta1*(np.pi/180) ), np.sin(theta1*(np.pi/180))], [np.cos(theta2*(np.pi/180)), np.sin(theta2*(np.pi/180))]])
        b = np.array([R_row1, R_row2])

        corner_pts = np.linalg.solve(A_matrix, b)
        pt = (int(corner_pts[0]), int(corner_pts[1]))

        if int(corner_pts[0])>0 and int(corner_pts[1]) > 0:
            corners.append(pt)
                                      
        cv2.circle(morph1_img, pt, 5, (0, 0, 0), -1)
 
    if len(corners) != 4:  
       continue     

    corners.sort(key = lambda point: point[1], reverse=True)

    recorded_paper_corners = corners

    real_paper_corners = np.array(([[0,0], [21.6, 0], [21.6, 27.9], [0, 27.9]]))

    A_matrix = []
    
    for i in range(4):
        x, y = recorded_paper_corners[i]
        u, v = real_paper_corners[i]
        A_matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A_matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        
    A_matrix = np.array(A_matrix)
    U, S, V = np.linalg.svd(A_matrix)
    H = V[-1, :].reshape((3, 3))
    H = H / H[2, 2]
   
    K = np.array([[1.38E+03 * (scale_percent / 100) , 0, 9.46E+02* (scale_percent / 100)],
                  [0, 1.38E+03 * (scale_percent / 100), 5.27E+02* (scale_percent / 100)],
                  [0, 0, 1]])  

    K_inv = np.linalg.pinv(K)
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = H[:, 2]
    req_lambda = 1 / np.linalg.norm(np.dot(K_inv, H1))

    R_row1 = req_lambda * np.dot(K_inv, H1)
    R_row2 = req_lambda * np.dot(K_inv, H2)
    R_row3 = np.cross(R_row1, R_row2)

    T_matrix = req_lambda * np.dot(K_inv, H3)

    camera_x_translation.append(T_matrix[0])
    camera_y_translation.append(T_matrix[1])
    camera_z_translation.append(T_matrix[2])

    R_matrix = np.column_stack((R_row1, R_row2, R_row3)) 

    P_camera = np.column_stack((np.dot(K, R_matrix), T_matrix))

    R_matrix = P_camera[:3, :3]
    
    sin_roll = R_matrix[2, 1]
    cos_roll = R_matrix[2, 2]
    roll = np.arctan2(sin_roll, cos_roll)

    sin_pitch = -R_matrix[2, 0]
    cos_pitch = np.sqrt(R_matrix[2, 1]**2 + R_matrix[2, 2]**2)
    pitch = np.arctan2(sin_pitch, cos_pitch)

    sin_yaw = R_matrix[1, 0]
    cos_yaw = R_matrix[0, 0]
    yaw = np.arctan2(sin_yaw, cos_yaw)
    
    # Conditions to remove outliers

    if roll > 1:
      roll_plot.append(roll)

    if pitch < 0.6:
      pitch_plot.append(pitch)

    if yaw < -2:
      yaw_plot.append(yaw)

    # Show the Morphed frame with detected lines and corners
    cv2.imshow("Morphed Frame", morph1_img)
    
    # Display Canny Edge Detection Image
    #cv2.imshow('Canny Edge Detection', edges)
    key = cv2.waitKey(10)
    
    if  key == ord('q'):
        break

# Release the video and close all windows
# cap.release()
# cv2.destroyAllWindows()

for i in range(0,147):
  frame_count.append(i+1)


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(frame_count, camera_x_translation, camera_y_translation, camera_z_translation)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Movement')
plt.show()

plt.plot(range(0,len(roll_plot)), roll_plot)
plt.xlabel('Frame')
plt.ylabel('Roll')
plt.title('Roll')
plt.show()

plt.plot(range(0,len(pitch_plot)), pitch_plot)
plt.xlabel('Frame')
plt.ylabel('Pitch')
plt.title('Pitch')
plt.show()

plt.plot(range(0,len(yaw_plot)), yaw_plot)
plt.xlabel('Frame')
plt.ylabel('Yaw')
plt.title('Yaw')
plt.show()

