# CTVT58
# SSA - Computer Vision Coursework - Due Fri 6/12/2019

import cv2
import os
import numpy as np
from math import ceil
from statistics import median


# STEREO DISPARITY VARIABLES
master_path_to_dataset = "../data/"  # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed
skip_forward_file_pattern = ""  # set to timestamp to skip forward to
crop_disparity = False  # display full or cropped disparity image
pause_playback = False  # pause until key press after each image

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

left_file_list = sorted(os.listdir(full_path_directory_left))
max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

# YOLO VARIABLES
classes_file = 'coco.names'
config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'
keep_processing = True
skip_forward_file_pattern = ""  # set to timestamp to skip forward to
crop_disparity = False  # display full or cropped disparity image
pause_playback = False  # pause until key press after each image

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

objects_in_scene_classes = []
objects_in_scene_distances = []
classes_checked = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]

classes = None
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres
windowName = 'YOLOv3 object detection: ' + weights_file
cv2.imshow(windowName, cv2.WINDOW_NORMAL)

max_disparity = 64


############################# GET DISPARITY #############################
#TODO Imrove disparity calculation (this might help in the marks $$$)
def calculate_stereo_disparity(image_left, image_right):
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

    grey_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    grey_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    # perform preprocessing - raise to the power, as this subjectively
    # appears to improve subsequent disparity calculation
    grey_left = np.power(grey_left, 0.75).astype('uint8')
    grey_right = np.power(grey_right, 0.75).astype('uint8')

    disparity = stereoProcessor.compute(grey_left, grey_right)

    dispNoiseFilter = 5  # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)
    return disparity_scaled
    return disparity


def calculate_median(arr):
    if arr and (median(arr) < 45):
        return str(round(median(arr),2)) + " m"


####################### GET COORDINATES OF THE BOX #######################


############################## DRAW THE BOX ##############################
def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    distance = 5
    # construct label
    label = '%s (%.2f) = %.2f m away' % (class_name, confidence, distance)
    if(class_name in classes_checked):
        objects_in_scene_classes.append(class_name)
        objects_in_scene_distances.append(distance)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
                         (left + round(1.5*labelSize[0]), top + baseLine),
                         (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

################ REMOVE BOUNDING BOXES WITH LOW CONFIDENCE ###############
def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]
    classIDs = []
    confidences = []
    boxes = []

    for result in results:
        for detection in result:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    classIDs_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIDs_nms.append(classIDs[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIDs_nms, confidences_nms, boxes_nms)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

####################### SETTING DARKNET VARIABLES ########################
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
output_layer_names = getOutputsNames(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

######################### RUNNING THE ANALYSIS ###########################
for filename_left in left_file_list:

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    print(full_path_filename_left)
    print(full_path_filename_right)
    print()

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        backup_imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

        # cv2.imshow("Right Image", imgR)
        print("-- files loaded successfully")
        print()

        #TODO preprocessing of the left and right images to ignore bits (e.g car)
        # left mask
        # right mask
        disparity = calculate_stereo_disparity(imgL, imgR)
        
        cv2.imshow("Disparity", (disparity * (256. / max_disparity)).astype(np.uint8))

        frame = imgL
        start_t = cv2.getTickCount()

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # set the input to the CNN network
        net.setInput(tensor)
        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # distance = median(z_axis_value)
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50))

        objects_in_scene_classes = []
        objects_in_scene_distances = []

        cv2.imshow(windowName, frame)
        
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000
        print("YOLOv3 took %.2f ms to process this frame"% (stop_t))

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break  # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord('c')): # crop
            crop_disparity = not(crop_disparity)
        elif (key == ord('p')):  # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

cv2.destroyAllWindows()
quit()
