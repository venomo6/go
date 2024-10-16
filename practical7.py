import cv2
import numpy as np
img = cv2.imread('cat.jpg')
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharp_img = cv2.filter2D(img,-1,kernel=kernel)
gaussian_3=cv2.GaussianBlur(img,(0,0),2.0)
unsharp_img = cv2.addWeighted(img,1.5,gaussian_3, -0.5,0,img)
result = np.hstack((img,sharp_img, unsharp_img))
cv2.imshow('Result', result)
