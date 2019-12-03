# CTVT58
# November 2019
# SSA - Computer Vision Coursework - Due Fri 6/12/2019
# Main program

import sys

import numpy as np  # NOQA

import cv2 as cv

import math


# Setup of masterpath directory
master_path_to_dataset = ""

# If user hasn't specified, use default directory
try:
    master_path_to_dataset = sys.argv[1]
except IndexError:
    master_path_to_dataset = "../data/"


# Trackbar callback function - Details in yolo.py
def on_trackbar(val):
    return


# Draw the predicted bounding box on the specified image
# Details in yolo.py
def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    cv.rectangle(image, (left, top), (right, bottom), colour, 3)
    label = '%s:%.2f' % (class_name, confidence)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(image, (left, top - round(1.5*labelSize[1])),
                        (left + round(1.5*labelSize[0]), top + baseLine),
                        (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
# Details in yolo.py
def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

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

    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    return (classIds_nms, confidences_nms, boxes_nms)


# Get the names of the output layers of the CNN network
# Details in yolo.py
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


img = cv.imread("messi5.jpg", 0)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

quit()
