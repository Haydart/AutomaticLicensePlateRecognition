import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 150
subplot_width = 3
subplot_height = 5


def plot_image(img, subplot_index, title='', fix_colors=True):
    plt.subplot(subplot_height, subplot_width, subplot_index)

    if fix_colors:
        if len(img.shape) == 3 and img.shape[2] == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.title(title)
    plt.axis('off')


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 32, 32, 32)


def equalize_histogram(img):
    return cv2.equalizeHist(img)


def morphological_opening(img):
    opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=opening_mask, iterations=15)
    return cv2.subtract(img, opening_image)


def canny_edge_detection(img):
    return cv2.Canny(img, 170, 200)


if __name__ == '__main__':
    image = cv2.imread('./license_plate_snapshots/test_088.jpg')
    # image = imutils.resize(image, width=512)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_image(grayscale_image, 1, 'Original image grayscale')
    plot_image(grayscale_image, 2, 'Original image grayscale BGR', fix_colors=False)
    plot_image(canny_edge_detection(grayscale_image), 3, 'Canny on original image')

    noise_removed_image = bilateral_filter(grayscale_image)
    plot_image(noise_removed_image, 4, 'Bilateral filtering')
    noise_removed_image = bilateral_filter(grayscale_image)
    plot_image(noise_removed_image, 5, 'Bilateral filtering BGR', fix_colors=False)
    plot_image(canny_edge_detection(noise_removed_image), 6, 'Canny after bilateral')

    histogram_equalized_image = equalize_histogram(noise_removed_image)
    plot_image(histogram_equalized_image, 7, 'Histogram equalization')
    plot_image(histogram_equalized_image, 8, 'Histogram equalization BGR', fix_colors=False)
    plot_image(canny_edge_detection(histogram_equalized_image), 9, 'Canny after histogram equalization')

    subtracted_image = morphological_opening(histogram_equalized_image)
    plot_image(subtracted_image, 10, 'Opening subtracted')
    plot_image(subtracted_image, 11, 'Opening subtracted BGR', fix_colors=False)
    plot_image(canny_edge_detection(subtracted_image), 12, 'Canny after opening subtraction')

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    fig = plt.gcf()
    fig.set_size_inches(10, 15)

    print('calculated')

    plt.show()
