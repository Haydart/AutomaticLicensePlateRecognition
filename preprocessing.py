import cv2
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

histogram_equalized_image = cv2.equalizeHist(noise_removed_image)
plt.subplot(2, 2, 3)
plt.imshow(histogram_equalized_image)
plt.title('histogram equalization')

edges_image = cv2.Canny(histogram_equalized_image, 170, 200)
plt.subplot(2, 2, 4)
plt.imshow(edges_image)
plt.title('canny edge detector')

plt.subplots_adjust(bottom=0.05, left=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.3)
plt.show()
