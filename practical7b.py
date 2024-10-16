import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('cat.jpg')
kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img)
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title("smooth image")
plt.xticks([]),plt.yticks([])
plt.show()
result = np.hstack((dst,img))
cv2.imshow('Result', result)

