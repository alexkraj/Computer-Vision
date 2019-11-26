# A Krajewski
# November 2019
# SSA - Computer Vision Coursework
# Main program

import numpy as numpy # NOQA
import cv2 as cv

img = cv.imread("messi5.jpg", 0)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
quit()
