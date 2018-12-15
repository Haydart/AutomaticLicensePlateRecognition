from random import randint

from utils import *


def approximate_contours(preprocessed_image, original_image):
    _, contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # take into consideration only 10 contours covering greatest area

    possible_polygons_with_areas = []  # [(contour, area), (...)]

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        print("Contour area: ", contour_area, " contour is ", contour)
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour_polygon = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, closed=True)
        print("app contour polygon", approximated_contour_polygon)
        possible_polygons_with_areas.append((approximated_contour_polygon, contour_area))

    result_polygon = max(possible_polygons_with_areas, key=lambda item: item[1])[0]

    for polygon, _ in possible_polygons_with_areas:
        original_image = draw_plate_polygons(original_image, polygon)

    # for contour in contours:
    #     original_image = draw_plate_contour(original_image, contour)

    return original_image, result_polygon


def draw_plate_polygons(image, approximated_polygon):
    return cv2.drawContours(image, [approximated_polygon], -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 3)


def order_points(points):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    sum = points.sum(axis=1)
    rect[0] = points[np.argmin(sum)]
    rect[2] = points[np.argmax(sum)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    warp_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, warp_matrix, (max_width, max_height))

    # return the warped image
    return warped


def process():
    image = load_image('skewed_trimmed_samples/skewed_004.jpg')
    image_gray = gray_scale(image)
    ret, binarized_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    eroded_image = erosion(binarized_image)
    closed_image = morphological_closing(eroded_image, (3, 3), iterations=4)

    preprocessed_image_with_polygons, _ = approximate_contours(closed_image, image)

    cv2.imshow("output", closed_image)
    # cv2.imshow("Preprocessed image", closed_image)
    # cv2.imshow("Contours on preprocessed image", preprocessed_image_with_contours)
    cv2.imshow("output2", preprocessed_image_with_polygons)
    # cv2.imwrite("output/polygons_on_dilated_photo4.jpg", preprocessed_image_with_polygons)

    cv2.waitKey()


    warped = four_point_transform(image, pts)

    # show the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)


if __name__ == '__main__':
    process()
