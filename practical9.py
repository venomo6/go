import cv2
import numpy as np
image = cv2.imread('cat.jpg')
kernel = np.ones((5,5),np.uint8)
cv2.imshow('input', image)
cv2.imshow('Erosion', cv2.erode(image,kernel,iterations=1))
cv2.imshow('dilation', cv2.dilate(image,kernel,iterations=1))
cv2.imshow('opening', cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel))
cv2.imshow('closing', cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel))
cv2.imshow('gradient', cv2.morphologyEx(image,cv2.MORPH_GRADIENT,kernel))

