## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[image4]: ./output/analysis_images.png
[video1]: ./output/test_mapping.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
Each of the below images are thresholded for a different type of detecton: navigable terrain, obstacles, and samples.
I experimented with stacking them on top of each other to get a good feel of what i was doing, and in the last image
i use convert the coordinates to the robot's perspective.

![alt text][image4]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
Below is the link to the test_output video from my populated `process_image()` function. All the previously defined functions were implemnted in a sequence to form a pipeline to produce the below result 

![test_ouput][video1]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

First we define source and destination points from the grid image to transform 1m x 1m area to top view and scale it.
Then we threshold the images for each component (navigable terrain, obstacles, and samples), then warp them, convert their coordinates to robot coordinates and then to world coordinates in order to be able to map what the front camera sees.
Those pixels are then plotted with different colors to indicate which detection is which. All the functions supplied were used, but before that i wrote them as i was following the lectures in order to fully grasp them and modify them as i want.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

I improved the angle at which the robot navigates by adding a function `steering()` in the `decision.py` file. It tries to center the robot in the navigable terrain by trying to get the mean angle of navigable terrain to 0. That improved its performance. The rover fails though when it hits the obstacles in the middle of the map. Maybe a different detection for those specific kind of obstacles would be a possible solution.


Simulator Settings: Fastest @ 1024 x 768

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**


**Describe in your writeup (and identify where in your code) how you modified or added functions to add obstacle and rock sample identification.**

For rock sample identification, i implemented a function similar to the one found in the opencv documentation referenced in the Udacity lectures. It basically masks the image in order to leave out the yellow colored pixels based on the HSV colorspace, then the image is converted to a binary representation to extract the pixels of the sample. Now that we have the x and y pixels of the sample, we can move on to convert them to rover coordinates and then world coordinates in order to map them. The function (implemented in both the test notebook and the pereception.py file) is as follows:
```python
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
```

As for Obstacle identification, i used the same provided function `color_thresh()` but instead of just searching for pixels that are only above the threshold, i modified it to search for the pixels that are bigger than or equal the threshold, so as to provide it with a (0, 0, 0) threshold tuple to include everything in the image, then i used opencv's `cv2.subtract()` function to subtract the navigable terrain pixels from the image. It was hinted that it would work fine if i chose the navigable terrain pixels, and then designated the obstacles to be everything else. Also used in both the test notebook and the pereception.py file.

```python
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
```    


**Describe in your writeup how you modified the process_image() to demonstrate your analysis and how you created a worldmap. Include your video output with your submission.**

First, rover front cam input image was thresholded for the sample and navigable terrain pixels. The image is also thresholded with (0, 0, 0) as threshold values in order to convert all of it binary, then the navigable pixels are subtracted to leave us with the obstacles (assuming everything else is an obstacle).
```python
    color_select_rock = detect_yellow(img)
    color_select = color_thresh(img, rgb_thresh)
    # Here i subtract the navigable terrain from the whole image by using cv2.subtract to also account for
    # negative values (if there are any).
    color_select_obstacles = cv2.subtract(color_thresh(img, (0, 0, 0)), color_select)
```    

Each of the thresholded images is warped using the provided function `perspect_transform()` using predefine source and destination points, previously calibrated during testing on the grid image.
```python
    warped_color_select_obstacles = perspect_transform(color_select_obstacles, src_pts, dst_pts)
    warped_color_select_rock = perspect_transform(color_select_rock, src_pts, dst_pts)
    warped_color_select = perspect_transform(color_select, src_pts, dst_pts)
```    

Next, pixels for each detection are transformed to rover coordinates and then world coordinates using the functions provided.

```python
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
```

Here we ignore appending the pixels to the world map if the robot's roll or pitch are not close enough to zero.

```python
    # Thresholding to remove false terrain identification
    if ((Rover.pitch > 359.5) or (Rover.pitch < 0.5)) & ((Rover.roll > 359.5) or (Rover.roll < 0.5)):
            # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[obstacles_world_y, obstacles_world_x, 0] += 1
            #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] +=1
            #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
Rover.worldmap[y_world, x_world, 2] += 1
```

Finally we convert the extracted information to navigable angles and distances to be used in the decision step
```python
    # 8) Convert rover-centric pixel positions to polar coordinates
    r, theta = to_polar_coords(x_rover, y_rover)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    Rover.nav_dists = r
Rover.nav_angles = theta
```

A video for the output of the pipeline: 

![test_ouput][video1]
