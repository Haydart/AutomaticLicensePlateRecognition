from PIL import Image
from pytesseract import pytesseract

from util.basic_transformations import BasicTransformations
from util.image_display_helper import ImageDisplayHelper

dh = ImageDisplayHelper(True, 2, 20)
bt = BasicTransformations(dh)


def process_image(image):
    binarized_image = bt.otsu_threshold(image)
    config = '-l eng --oem 1 --psm 10'
    ocr_string = pytesseract.image_to_string(Image.fromarray(binarized_image), config=config)
    cleaned_ocr_string = ''.join([char for char in ocr_string if char.isupper() or char.isdigit()])

    if len(cleaned_ocr_string) >= 5 and any(char.isdigit() for char in cleaned_ocr_string):
        print(cleaned_ocr_string)
