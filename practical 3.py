import cv2
import numpy as np
img = cv2.imread('cat.jpg')
blur= np.ones((5,5), np.float32)/25
image_filter_b = cv2.filter2D(src=img, ddepth=-1, kernel =blur)
sharp= np.array([[0, -1, 0],[-1,5,-1],[0, -1, 0]])
image_filter_s = cv2.filter2D(src=img, ddepth=-1, kernel =sharp)
cv2.imshow("original",img)
cv2.imshow("sharpend",image_filter_s)
cv2.imshow("blur",image_filter_s)

