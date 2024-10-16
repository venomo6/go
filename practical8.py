import cv2
import numpy as np
img=cv2.imread('cat.jpg')
img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gry,(3,3),0)
sobelx=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=5)
sobely=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=5)
sobelxy=cv2.Sobel(src=img_blur,ddepth=cv2.CV_64F,dx=1,dy=1,ksize=5)
result = np.hstack((sobelx,sobely, sobelxy))
cv2.imshow('Result,Sobel of (X,Y,XY)', result)
edges = cv2.Canny(image=img_blur,threshold1 = 100,threshold2=200)
cv2.imshow('Canny Edge',edges)

