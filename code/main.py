# CTVT58
# November 2019
# SSA - Computer Vision Coursework - Due Fri 6/12/2019
# Main program

import cv2
import os
import numpy as np
from math import ceil
import argparse
import sys

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
args = parser.parse_args()

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

classes = None
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000  # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres
windowName = 'YOLOv3 object detection: ' + weights_file
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

max_disparity = 64;

# YOLO METHODS
def on_trackbar(val):
    return


def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f is %.2f meters away' % (class_name, confidence, 5)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
                         (left + round(1.5*labelSize[0]), top + baseLine),
                         (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


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
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# STARTING UP VARIABLE WORK:

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
output_layer_names = getOutputsNames(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# RUNNING OF THE SIMULATION:
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

        cv2.imshow("Right Image", imgR)
        print("-- files loaded successfully")
        print()

        # while (keep_processing):
        start_t = cv2.getTickCount()
        frame = imgL

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # set the input to the CNN network
        net.setInput(tensor)
        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        # confThreshold = cv2.getTrackbarPos(trackbarName, windowName) / 100
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50))

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        newName = "image"
        cv2.imshow(newName, frame)
        # cv2.setWindowProperty(newName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN & args.fullscreen)

        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # start the event loop + detect specific key strokes
        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, 40 - int(ceil(stop_t)))) & 0xFF

        if (key == ord('x')):
            keep_processing = False

        # imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        # cv2.imshow('Right Image', imgR)

       
        key = cv2.waitKey(max(2, 40 - int(ceil(stop_t)))) & 0xFF
        # key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break  # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("left.png", imgL)
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

cv2.destroyAllWindows()
