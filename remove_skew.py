from utils import *


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def find_contours(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For this problem, number plate should have contours with a small area as compared to other contours.
    # Hence, we sort the contours on the basis of contour area and take the least 10 contours
    return sorted(contours, key=cv2.contourArea, reverse=True)[:10]


def approximate_contour(contours):
    possible_polygons_with_perimeters = []  # [(contour, perimeter), (...)]

    for contour in contours:
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour_polygon = cv2.approxPolyDP(contour, 0.05 * contour_perimeter, closed=True)
        possible_polygons_with_perimeters.append((approximated_contour_polygon, contour_perimeter))

    result_polygon = max(possible_polygons_with_perimeters, key=lambda item: item[1])[0]
    return result_polygon


def draw_plate_contour(img, approximated_polygon):
    return cv2.drawContours(img, [approximated_polygon], -1, (0, 255, 0), 3)


if __name__ == '__main__':
    image = load_image('skewed_trimmed_samples/skewed_001.jpg')
    image_gray = gray_scale(image)
    ret, binarized_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    eroded_image = erosion(binarized_image)
    closed_image = morphological_closing(eroded_image)
    im2, contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    approximated_polygon = approximate_contour(contours)
    final_image = draw_plate_contour(image, approximated_polygon)

    cv2.imshow("Grayscale", image_gray)
    cv2.imshow("Binarized", binarized_image)
    cv2.imshow("Binarized eroded", eroded_image)
    cv2.imshow("Warped", image)
    cv2.imshow("Morphological closing", closed_image)

    cv2.waitKey()

    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", help="path to the image file")
    # ap.add_argument("-c", "--coords",
    #                 help="comma seperated list of source points")
    # args = vars(ap.parse_args())
    #
    # # load the image and grab the source coordinates (i.e. the list of
    # # of (x, y) points)
    # # NOTE: using the 'eval' function is bad form, but for this example
    # # let's just roll with it -- in future posts I'll show you how to
    # # automatically determine the coordinates without pre-supplying them
    # image = cv2.imread(args["image"])
    # pts = np.array(eval(args["coords"]), dtype="float32")
    #
    # # apply the four point tranform to obtain a "birds eye view" of
    # # the image
    # warped = four_point_transform(image, pts)
    #
    # # show the original and warped images
    # cv2.imshow("Original", image)
    # cv2.imshow("Warped", warped)
    # cv2.waitKey(0)
