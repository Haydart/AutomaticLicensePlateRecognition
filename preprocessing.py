import cv2
import imutils
import matplotlib.pyplot as plt

image = cv2.imread('./snapshots/test_001.jpg')
# image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 2, 1)
plt.imshow(grayscale_image, 'gray')
plt.title('grayscale')

noise_removed_image = cv2.bilateralFilter(grayscale_image, 9, 75, 75)
plt.subplot(2, 2, 2)
plt.imshow(noise_removed_image)
plt.title('bilateral filter')

edges_image = cv2.Canny(noise_removed_image, 170, 200)
plt.subplot(1, 3, 3)
plt.imshow(edges_image)
plt.title('canny edge detector')

plt.show()
