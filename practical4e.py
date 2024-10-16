import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('cat.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img, 120,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120,255,cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120,255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120,255,cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120,255,cv2.THRESH_TOZERO_INV)
result = np.hstack((thresh1, thresh2, thresh3, thresh4, thresh5))
cv2.imwrite('Thresholding.png', result)

