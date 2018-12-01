import cv2
import imutils
import matplotlib.pyplot as plt

subplot_width = 3
subplot_height = 5


def plot_image(image, subplot_index, title='', gray=False):
    plt.subplot(subplot_height, subplot_width, subplot_index)
    plt.imshow(image, cmap='gray' if gray else None)
    plt.title(title)
    plt.axis('off')


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 200, 30, 30)


def equalize_histogram(img):
    return cv2.equalizeHist(img)


def morphological_opening(img):
    opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening_image = cv2.morphologyEx(histogram_equalized_image, cv2.MORPH_OPEN, kernel=opening_mask, iterations=15)
    return cv2.subtract(img, opening_image)


def canny_edge_detection(img):
    return cv2.Canny(img, 170, 200)


if __name__ == '__main__':
    image = cv2.imread('./snapshots/test_001.jpg')
    image = imutils.resize(image, width=400)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_image(grayscale_image, 1, 'Grayscale', gray=True)

    noise_removed_image = bilateral_filter(grayscale_image)
    plot_image(noise_removed_image, 2, 'Smoothing', gray=True)

    histogram_equalized_image = equalize_histogram(noise_removed_image)
    plot_image(histogram_equalized_image, 3, 'Histogram equalization')

    subtracted_image = morphological_opening(histogram_equalized_image)
    plot_image(subtracted_image, 4, 'Subtracted opening')

    canny_edge_image = canny_edge_detection(subtracted_image)
    plot_image(canny_edge_image, 5, 'Canny edge detection')

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    fig = plt.gcf()
    fig.set_size_inches(12, 18)

    plt.show()
