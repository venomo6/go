import cv2
import numpy as np
from skimage.feature import hog
img = cv2.imread("cat.jpeg")
img1 = cv2.imread("cat.jpeg")
hog,_image = hog(img, orientations = 8, pixels_per_cell=(16,16),cells_per_block=(1,1),
visualize= True, multichannel=True)
result = np.hstack((img,img1))
cv2.imshow('HOG image,Original', result)
cv2.imwrite("output1s.jpg",img)
