import cv2
import numpy as np
noisy_img = cv2.imread('cat.jpg',1)

size = 5
denoise_img = cv2.medianBlur(noisy_img,size)

result = np.hstack((noisy_img, denoise_img))
cv2.imwrite('Non Linear smoothing - median filter.jpg', result)
cv2.imshow('Result', result)

