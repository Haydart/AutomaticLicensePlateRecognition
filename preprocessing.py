import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./snapshots/test_001.jpg')
# image = imutils.resize(image, width=500)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(3, 2, 1)
plt.imshow(grayscale_image, 'gray')
plt.title('grayscale')
plt.axis('off')

noise_removed_image = cv2.bilateralFilter(grayscale_image, 200, 17, 17)
plt.subplot(3, 2, 2)
plt.imshow(noise_removed_image)
plt.title('bilateral filter')
plt.axis('off')

histogram_equalized_image = cv2.equalizeHist(noise_removed_image)
plt.subplot(3, 2, 3)
plt.imshow(histogram_equalized_image)
plt.title('histogram equalization')
plt.axis('off')

openingMask = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morphological_opening_mask = cv2.morphologyEx(noise_removed_image, cv2.MORPH_OPEN, kernel=openingMask, iterations=15)
plt.subplot(3, 2, 4)
plt.imshow(morphological_opening_mask)
plt.title('morphological opening')
plt.axis('off')

edges_image = cv2.Canny(histogram_equalized_image, 170, 200)
plt.subplot(3, 2, 5)
plt.imshow(edges_image)
plt.title('canny edge detector')
plt.axis('off')


plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

fig = plt.gcf()
fig.set_size_inches(12, 18)

plt.show()
