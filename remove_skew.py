from random import randint

from utils import *


def find_plate_contour(preprocessed_image, original_image):
    _, contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # take into consideration only 10 contours covering greatest area

    polygons_with_areas = []  # [(contour, area), (...)]

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        print("Contour area: ", contour_area, " contour is ", contour)
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour_polygon = cv2.approxPolyDP(contour, 0.03 * contour_perimeter, closed=True)
        print("approximated contour polygon", approximated_contour_polygon)
        polygons_with_areas.append((approximated_contour_polygon, contour_area))

    result_polygon = max(polygons_with_areas, key=lambda item: item[1])[0]
    return original_image, result_polygon


def draw_plate_polygons(image, approximated_polygon):
    return cv2.drawContours(image, [approximated_polygon], -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 3)


def order_corner_points(points):
    # initialize a list of coordinates that will be ordered top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    ooints_sum = points.sum(axis=1)
    rect[0] = points[np.argmin(ooints_sum)]
    rect[2] = points[np.argmax(ooints_sum)]
    # top-right point will have the smallest difference, bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(image, points):
    rect = order_corner_points(points)
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

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    warp_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, warp_matrix, (max_width, max_height))

    return warped


def process():
    image = load_image('skewed_trimmed_samples/skewed_003.jpg')
    gray_image = gray_scale(image)

    ret, binarized_image = cv2.threshold(gray_image, 175, 255, cv2.THRESH_BINARY)
    eroded_image = erosion(binarized_image)
    closed_image = morphological_closing(binarized_image, iterations=5)
    eroded_closed_image = morphological_closing(eroded_image, iterations=5)

    _, result_polygon = find_plate_contour(eroded_closed_image, image)
    polygon_flat_list = [item for sublist in result_polygon for item in sublist]
    plate_corners_list = [(arr[0], arr[1]) for arr in polygon_flat_list]

    deskewed_image = four_point_transform(image, np.array(plate_corners_list))
    draw_plate_polygons(image, result_polygon)

    cv2.imshow("Grayscale", gray_image)
    cv2.imshow("Bin", binarized_image)
    cv2.imshow("Bin -> Erosion", eroded_image)
    cv2.imshow("Bin -> Erosion -> Closing", eroded_closed_image)
    cv2.imshow("Bin -> Closing", closed_image)
    cv2.imshow("Result Polygon", image)
    cv2.imshow("Deskewed image", deskewed_image)

    cv2.waitKey()


if __name__ == '__main__':
    process()
