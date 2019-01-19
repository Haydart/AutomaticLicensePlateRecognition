import glob
import cv2


class ImageLoader:

    """Loads images from source file.

    Args:
        file_extension (list): List of file extensions strings to be loaded
    """

    def __init__(self, file_extension=['jpg', 'jpeg', 'png']):
        self.file_extension = file_extension
        self.patterns = ['*.{}'.format(ext) for ext in self.file_extension ]

    def load_images(self, source_path):
        """
        Generate images from source.

        :param source_path:
        :return: image: Image in form of matrix
        """

        for file in self.__get_all_paths(source_path):
            yield self.__load_image(file)

    def __get_all_paths(self, source_path):
        paths = ['{}{}'.format(source_path, pattern) for pattern in self.patterns]

        files = []
        for path in paths:
            files.extend(glob.glob(path))

        return files

    def __load_image(self, file):
        return cv2.imread(file)
