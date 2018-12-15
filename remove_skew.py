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
        approximated_contour_polygon = cv2.approxPolyDP(contour, 0.05 * contour_perimeter, closed=True)
        print("app contour polygon", approximated_contour_polygon)
        possible_polygons_with_areas.append((approximated_contour_polygon, contour_area))

    result_polygon = max(possible_polygons_with_areas, key=lambda item: item[1])[0]

    for contour in contours:
        original_image = draw_plate_contour(original_image, contour)

    return original_image, result_polygon


def draw_plate_contour(image, approximated_polygon):
    return cv2.drawContours(image, [approximated_polygon], -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 3)


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


def process():
    image = load_image('skewed_trimmed_samples/skewed_003.jpg')
    image_gray = gray_scale(image)
    ret, binarized_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    # eroded_image = erosion(binarized_image)
    # closed_image = morphological_closing(eroded_image)

    preprocessed_image_with_contours, _ = approximate_contours(binarized_image, image)
    # preprocessed_image_with_contours, _ = approximate_contours(closed_image, image)

    cv2.imshow("Binarized", binarized_image)
    # cv2.imshow("Preprocessed image", closed_image)
    # cv2.imshow("Contours on preprocessed image", preprocessed_image_with_contours)
    cv2.imshow("Contours on original image", preprocessed_image_with_contours)
    cv2.imwrite("output/contours_on_binarized_photo3.jpg", preprocessed_image_with_contours)

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


if __name__ == '__main__':
    process()
