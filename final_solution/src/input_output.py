import glob
import cv2
import os


class Image:

    def __init__(self, image, path):
        self.image = image
        self.path = path


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
        :return: image: Image
        """

        for file in self.__get_all_paths(source_path):
            yield Image(
                image=self.__load_image(file),
                path=file
            )

    def __get_all_paths(self, source_path):
        paths = ['{}{}'.format(source_path, pattern) for pattern in self.patterns]

        files = []
        for path in paths:
            files.extend(glob.glob(path))

        return files

    def __load_image(self, file):
        return cv2.imread(file)


class ImageSaver:

    def __init__(self, path):
        self.path = path

    def save_image(self, image, counter):
        path = self.__make_save_path(image.path, counter)
        cv2.imwrite(path, image.image)
        print('Image saved at:' + path)

    def __make_save_path(self, source_path, counter):
        source_name = source_path.split('/')[-1]

        name = source_name + str(counter) + source_name
        path = os.path.join(self.path, name)

        return path
