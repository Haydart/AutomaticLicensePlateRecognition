import cv2

from utils import gray_scale, load_image


def detect_vehicles_in_photo():
    image = load_image("dataset/track003.png")
    car_cascade = cv2.CascadeClassifier("cars.xml")

    gray = gray_scale(image)

    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("fucc", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    detect_vehicles_in_photo()
