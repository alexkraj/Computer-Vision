#####################################################################

# Example : load, display and compute SGBM disparity
# for a set of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import get_depth_from_disparity
from math import ceil
from statistics import median

############################### INITIALISE VALUES ######################################

class_file = 'coco.names'
config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'

classes = None
with open(class_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres
windowName = 'YOLOv3 object detection: ' + weights_file
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

max_disparity = 64;

# where is the data ? - set this to where you have it

master_path_to_dataset = "TTBB-durham-02-10-17-sub10";  # * need to edit this *
directory_to_cycle_left = "left-images";  # edit this if needed
directory_to_cycle_right = "right-images";  # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = "";  # set to timestamp to skip forward to

crop_disparity = False;  # display full or cropped disparity image
pause_playback = False;  # pause until key press after each image

# resolve full directory location of data set for left / right images
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right);

########################################################################################################################




# get a list of the left image files and sort them (by timestamp in filename)
def on_trackbar(val):
    return



#######################     PREPROCESS     ################ ################ ##############################################


#######################     DISPARITY     ################ ################ ##############################################


# Called each time we want to return ROI's and disparity map
def generate_stereo_disparity(imgL, imgR):
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY);

    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation

    grayL = np.power(grayL, 0.75).astype('uint8');
    grayR = np.power(grayR, 0.75).astype('uint8');

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(grayL, grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5;  # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);
    return disparity_scaled

    return disparity

# returns regions of interest that we want to keep
def get_region_of_interest(disparity):
    f = 399.9745178222656
    B = 0.2090607502
    height, width = disparity.shape[:2]
    # print(height, width)
    y_x = []
    z = []
    for y in range(height):
        for x in range(width):
            if (disparity[y,x] > 0):
                Z = (f * B) / disparity[y,x]
                if (Z>5 and Z<25):
                    y_x.append((y,x))
                    z.append(Z)
    return y_x


########################################### GET DISPARITY FROM BOUNDED BOX #############################################

def get_median(Z):
    # cur_median = median(Z)
    if Z and median(Z) < 45:
        return " " + str(round(median(Z),2)) + " m"    # statistics.median(z) #Median
    return ""

######################################## GET COORDINATES OF THE BOX ####################################################

def get_horizontal_center_of_bounding(a,b, pixels):  # Extract horizontal coordinates
    return ceil((a + b) // 2 - pixels), ceil((a + b) // 2 + pixels)


def get_vertical_center_of_bounding(a,b, pixels):   # Extract vertical coordinates
    return ceil((a + b) // 2 - pixels), ceil((a + b) // 2 + pixels)


def get_mean_pixels(top,bottom,left,right): # Get 30 % of the box in the middle
    return ((top - bottom) * (left - right) * 0.3)/100


def get_y_x_coordinates_array (top,bottom,left,right):  # Calculate box coordinates
    horzontal_start, horizontal_end = get_horizontal_center_of_bounding(left, right,
                                                                        get_mean_pixels(top, bottom, left, right)/2)
    vertical_start, vertical_end = get_vertical_center_of_bounding(top, bottom,
                                                                   get_mean_pixels(top, bottom, left, right)/2)
    # Go through the bounding boxes and put the coordinates in a resultant array
    return [[y, x] for x in range(horzontal_start, horizontal_end) for y in range(vertical_start, vertical_end)]


################################# DRAW THE BOX    ######################################################################

# Draw the predicted bounding box
def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f' % (class_name, confidence)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)



################################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# def pre_filtering(image):
#     # cv2.imshow('old: ' ,image)
#     # img_d = downsample(image)
#     img_denoised = denoise(image)
#     correct = correct_gamma(img_denoised, 0.2)
#     # bright_low = lower_brightness(correct)
#     return correct



left_file_list = sorted(os.listdir(full_path_directory_left));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
# stereoProcessor = cv2.StereoSGBM_create(
# 		minDisparity=0,
# 		numDisparities=max_disparity,
# 		P1=8 * 3 * (11 ** 2),
# 		P2=32 * 3 * (11 ** 2),
# 		blockSize=15,
# 		preFilterCap=63,
# 		disp12MaxDiff=3,
# 		uniquenessRatio=10,
# 		speckleRange=2,
# 		speckleWindowSize=100,
# 		mode=cv2.STEREO_SGBM_MODE_HH4
#     )


# applying disparity map post-filting
# from the open cv docs
# https://docs.opencv.org/3.4.2/d3/d14/tutorial_ximgproc_disparity_filtering.html
# creating a wls filter
# FILTER Parameters

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)
    if ((len(skip_forward_file_pattern) > 0) and not (skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames
    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        backup_image = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)


        cv2.imshow('right image', imgR)
        print("-- files loaded successfully");
        print();
        # PREFILTER HERE TODO  imgL prefilter
        left_mask = cv2.imread('Masks/left_image_mask.png', cv2.CV_8U)
        right_mask = cv2.imread('Masks/right_image_mask.png', cv2.CV_8U)
        disparity = generate_stereo_disparity(imgL, imgR)

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(imgL, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)


        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # Remove the bounding boxes with low confidence
        classIds, boxes, confidences, confThreshold, nmsThreshold = postprocess(imgL, results)
        
        # TODO REPORT THIS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # Calculate coordinates of x,y and select 30% in the middle
            horizontal_start, horizontal_end = get_horizontal_center_of_bounding(left, left+width,
                                                                                 get_mean_pixels(top, top + height, left,
                                                                                                 left + width) / 2)
            vertical_start, vertical_end = get_vertical_center_of_bounding(top, top + height,
                                                                           get_mean_pixels(top, top + height, left,
                                                                                           left + width) / 2)
            Z = []
            # Go through all pixels and calculate Z taking into account focal length and baseline
            for x in range(horizontal_start, horizontal_end):
                for y in range(vertical_start, vertical_end):
                    # Calculate Z = (f * 😎 / disparity[y, x];
                    try:
                        if (disparity[y, x] > 0):
                            Z.append((camera_focal_length_px * stereo_camera_baseline_m) / disparity[y, x])
                    except IndexError:
                        continue

            depth = get_median(Z)  # Get depth by calculating a median of 30% of the box pixels in the middle

            drawPred(classIds[i], confidences[i], left, top, left + width, top + height, backup_image,(255, 178, 50),depth)

        # postprocess(imgL, results, confThreshold, nmsThreshold)
        cv2.imshow("YOLO Object Detection using OpenCV", backup_image)
        # cv2.imwrite(filename_left,backup_image)
        cv2.waitKey(20)

    else:
        print("-- files skipped (perhaps one is missing or not PNG)");
        print();

# close all windows

cv2.destroyAllWindows()

#####################################################################