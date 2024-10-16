import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('cat.jpg')
contrast_start = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
channel, a, b = cv2.split(contrast_start)
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
cl = clahe.apply(channel)
limb = cv2.merge((cl,a,b))
enhanced_img = cv2.cvtColor(limb, cv2.COLOR_LAB2BGR)
result = np.hstack((image, enhanced_img))
cv2.imwrite('Contrast_adjustment.jpg', result)
cv2.imshow('Result', result)
