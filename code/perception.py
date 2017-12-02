import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only

def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] >= rgb_thresh[0]) \
                & (img[:,:,1] >= rgb_thresh[1]) \
                & (img[:,:,2] >= rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# Inspired by the opencv tutorials link provided in the Udacity lectures, this function detects yellow colors
# by converting to the HSV colorspace and thresholding for yellow, a mask is then applied to extract only the ROI

def detect_yellow(img):
    
    copy = np.copy(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    
    masked_rgb_img = cv2.bitwise_and(copy, copy, mask = yellow_mask)
    gray = cv2.cvtColor(masked_rgb_img, cv2.COLOR_RGB2GRAY)
    yellow_indices = gray[:, :] > 100
    
    binary_img = np.zeros_like(copy[:, :, 0])
    binary_img[yellow_indices] = 1

    return binary_img





# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


"""

def rover_coords_obstacles(binary_img_for_shape_purposes):
    
    # Bottom offset is hard-coded----Change later
    ypos, xpos = np.where(binary_img_for_shape_purposes[:-5, :] == 1)
 
    x_pixel = -(ypos - binary_img_for_shape_purposes.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img_for_shape_purposes.shape[1]/2 ).astype(np.float)
    
    return x_pixel, y_pixel
"""

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world



# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped




# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    image = Rover.img
    img = np.copy(image)
    dst_size = 5
    bottom_offset = 6
    scale = 10
    
    src_pts = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst_pts = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    rgb_thresh = (160, 160, 160)
    
    color_select_rock = detect_yellow(img)
    color_select = color_thresh(img, rgb_thresh)
    # Here i subtract the navigable terrain from the whole image by using cv2.subtract to also account for
    # negative values (if there are any).
    color_select_obstacles = cv2.subtract(color_thresh(img, (0, 0, 0)), color_select)
    
    # 2) Apply perspective transform
    warped_color_select_obstacles = perspect_transform(color_select_obstacles, src_pts, dst_pts)
    warped_color_select_rock = perspect_transform(color_select_rock, src_pts, dst_pts)
    warped_color_select = perspect_transform(color_select, src_pts, dst_pts)


    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    Rover.vision_image[:,:,0] = warped_color_select_obstacles * 255
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    Rover.vision_image[:,:,1] = warped_color_select_rock * 255
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = warped_color_select * 255
   
    
    
    
    
    # 5) Convert map image pixel values to rover-centric coords
    
    x_rover, y_rover = rover_coords(warped_color_select)
    obstacles_rover_x, obstacles_rover_y = rover_coords(warped_color_select_obstacles)
    rock_rover_x, rock_rover_y = rover_coords(color_select_rock)
    
    # 6) Convert rover-centric pixel values to world coordinates
    
    x_world, y_world = pix_to_world(x_rover, y_rover, Rover.pos[0], Rover.pos[1], 
                                   Rover.yaw, Rover.worldmap.shape[0], scale)
    
    rock_x_world, rock_y_world = pix_to_world(rock_rover_x, rock_rover_y, Rover.pos[0], Rover.pos[1], 
                                                   Rover.yaw, Rover.worldmap.shape[0], scale)
    
    obstacles_world_x, obstacles_world_y = pix_to_world(obstacles_rover_x, obstacles_rover_y, 
                                                       Rover.pos[0], Rover.pos[1], 
                                                       Rover.yaw, Rover.worldmap.shape[0], scale)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    
    # Thresholding to remove false terrain identification
    if ((Rover.pitch > 359.5) or (Rover.pitch < 0.5)) & ((Rover.roll > 359.5) or (Rover.roll < 0.5)):
            # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[obstacles_world_y, obstacles_world_x, 0] += 1
            #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] +=1
            #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        Rover.worldmap[y_world, x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    r, theta = to_polar_coords(x_rover, y_rover)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    Rover.nav_dists = r
    Rover.nav_angles = theta
    
    
    return Rover