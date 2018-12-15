import glob
from enum import Enum
import pandas as pd
import utils

class Dataset(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'


class DatasetsProvider:

    def __init__(self, source_path):
        self.path = source_path

    def load_train(self, img_ext='.png'):
        train_directory_path = self.path + 'training' + '/**/*'
        images = sorted(glob.glob(train_directory_path + img_ext))
        labels = sorted(glob.glob(train_directory_path + '.txt'))

        df = pd.DataFrame({'image': images, 'label': labels})
        return df

    def images(self):
        df = self.load_train()
        for index, row in df.iterrows():
            image = utils.load_image(row.image)
            label = self._label_file_to_dict(row.label)
            plate_position = label['position_plate'].strip()
            plate_number = label['plate'].strip()

            yield (image, plate_position, plate_number)

    def _label_file_to_dict(self, path):
        d = {}
        with open(path) as file:
            for line in file:
                (key, val) = line.split(':')
                d[key] = val
        return d


if __name__ == '__main__':
    dp = DatasetsProvider(
        source_path='/home/lukasz/Studia/Analiza obrazow i wideo/UFPR-ALPR dataset/'
    )

    for example in dp.images():
        print(example)
