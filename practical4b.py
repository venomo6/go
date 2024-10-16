import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('cat.jpg')
for i in [0.1, 0.5, 1.2, 2.2]:
 gamma_image = np.array(255*(image / 255) ** i)
 cv2.imwrite('gamma_transformed'+str(i)+'.jpg', gamma_image)

