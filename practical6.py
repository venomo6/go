import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
img = cv2.imread('cat.jpg')
image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
figure_size = 9
gauss= np.random.normal(0,1,image.size)
gauss = gauss.reshape(image.shape[0],image.shape[1], image.shape[2]).astype('uint8')
img_gauss = cv2.add(image,gauss)
new_image= cv2.GaussianBlur(image,(figure_size,figure_size),0)
plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_gauss,cv2.COLOR_HSV2RGB))
plt.title('Gaussian Noise')
plt.xticks([]),plt.yticks([])
plt.show()
plt.subplot(121),
plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)),
plt.title('Gaussian Filter')
plt.xticks([]),plt.yticks([])
plt.show()

