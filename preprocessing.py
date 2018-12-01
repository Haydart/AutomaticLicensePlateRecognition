import cv2
import imutils
import matplotlib.pyplot as plt

subplot_width = 2
subplot_height = 5


def plot_image(image, subplot_index, title=''):
    plt.subplot(subplot_height, subplot_width, subplot_index)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')


if __name__ == '__main__':
    image = cv2.imread('./snapshots/test_001.jpg')
    image = imutils.resize(image, width=500)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_image(grayscale_image, 1, 'Grayscale')

    noise_removed_image = cv2.bilateralFilter(grayscale_image, 200, 17, 17)
    plot_image(noise_removed_image, 2, 'Noise removed')

    histogram_equalized_image = cv2.equalizeHist(noise_removed_image)
    plot_image(histogram_equalized_image, 3, 'Histogram equalization')

    opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening_image = cv2.morphologyEx(histogram_equalized_image, cv2.MORPH_OPEN, kernel=opening_mask, iterations=15)
    subtracted_image = cv2.subtract(histogram_equalized_image, opening_image)
    plot_image(subtracted_image, 4, 'Subtracted opening')

    canny_edge_image = cv2.Canny(noise_removed_image, 200, 256)
    plot_image(canny_edge_image, 5, 'Canny edge detection')

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    fig = plt.gcf()
    fig.set_size_inches(10, 15)

    plt.show()
