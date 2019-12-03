# CTVT58
# November 2019
# SSA - Computer Vision Coursework - Due Fri 6/12/2019
# Main program

import sys

import numpy as numpy  # NOQA

import cv2 as cv


# Setup of masterpath directory
master_path_to_dataset = ""

# If user hasn't entered, use default directory
try:
    master_path_to_dataset = sys.argv[1]
except IndexError:
    master_path_to_dataset = "../data/"

img = cv.imread("messi5.jpg", 0)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

quit()
