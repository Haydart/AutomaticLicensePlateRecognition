from PIL import Image
from pytesseract import pytesseract

if __name__ == '__main__':
    print(pytesseract.get_tesseract_version())
    # print(pytesseract.image_to_data(Image.open('ocr.png')))
    print(pytesseract.image_to_data(Image.open('main_pipeline/to_ocr4.jpg')))


def read_text(image_path):
    print(pytesseract.image_to_data(Image.open(image_path)))