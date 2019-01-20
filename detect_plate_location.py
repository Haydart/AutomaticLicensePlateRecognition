from util.image_display_helper import *
from util.input_output import *
from util.utils import *

display = ImageDisplayHelper(True, 3, 5)

if __name__ == '__main__':
    image = cv2.imread('dataset/license_plate_snapshots/test_002.jpg')
    # image = imutils.resize(image, width=512)

    grayscale_image = gray_scale(image)

    display.add_to_plot(grayscale_image, 1, 'Original image grayscale')
    display.add_to_plot(grayscale_image, 2, 'Original image grayscale BGR', fix_colors=False)

    vertical_image = sobel_vertical_edge_detection(grayscale_image)
    skeletonized_image, _ = skeletonization(vertical_image)
    display.add_to_plot(skeletonized_image, 3, 'Vertical Sobel -> skeletonization')

    noise_removed_image = bilateral_filter(grayscale_image)
    display.add_to_plot(noise_removed_image, 4, 'Bilateral filtering')
    display.add_to_plot(noise_removed_image, 5, 'Bilateral filtering BGR', fix_colors=False)

    vertical_image = sobel_vertical_edge_detection(noise_removed_image)
    skeletonized_vertical_edges_image = otsu_threshold(vertical_image)
    display.add_to_plot(skeletonized_vertical_edges_image, 6, 'bilateral -> VSobel -> skeleton')

    histogram_equalized_image = histogram_equalization(noise_removed_image)
    display.add_to_plot(histogram_equalized_image, 7, 'Histogram equalization')
    display.add_to_plot(histogram_equalized_image, 8, 'Histogram equalization BGR', fix_colors=False)
    display.add_to_plot(canny_edge_detection(histogram_equalized_image), 9, 'Canny after histogram equalization')

    subtracted_image = morphological_opening(histogram_equalized_image)
    display.add_to_plot(subtracted_image, 10, 'Opening subtracted')
    binarized_subtraction = otsu_threshold(subtracted_image)
    display.add_to_plot(binarized_subtraction, 11, 'Subtraction -> otsu', fix_colors=False)
    display.add_to_plot(otsu_threshold(sobel_vertical_edge_detection(binarized_subtraction)), 12,
                        'subtraction -> otsu -> sobel')

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    fig = plt.gcf()
    fig.set_size_inches(10, 15)

    print('calculated')

    plt.show()
