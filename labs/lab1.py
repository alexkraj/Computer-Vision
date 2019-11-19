import cv2 as cv
image = cv.imread('messi5.jpg',0)
cv.imshow('image',image)
cv.waitKey(0)
cv.destroyAllWindows()
