import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('cat.jpg')
equ = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
res = np.hstack((image, equ))
cv2.imshow('Equilize Histogram',res)

