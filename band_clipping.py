import numpy as np
from scipy import signal
import utils

mask_0 = [1, 3, 5, 7, 5, 9, 3, 1]
mask_1 = [1, 5, 9, 12, 15, 12, 9, 5, 1]
mask_2 = [4, 7, 16, 26, 41, 26, 16, 7, 4]
mask_3 = [.006, .061, .242, .383, .242, .061, .006]
mask_4 = [1, 4, 7, 16, 26, 41, 26, 16, 7, 4, 1]
mask_5 = [.000229, .005977,	.060598, .241732, .382928, .241732, .060598, .005977, .000229]
mask_6 = [1, 4, 16, 24, 36, 24, 16, 4, 1]
mask_7 = [2, 5, 8, 16, 20, 24, 30, 37, 30, 24, 20, 16, 8, 5, 2]
mask_8 = [2, 5, 8, 16, 20, 24, 30, 37, 42, 37, 30, 24, 20, 16, 8, 5, 2]


class BindsFinder:

    def __init__(self, image):
        self.image = np.array(image / np.max(image))
        self.mask = mask_8
        self.y_c = 0.55

    def _find_band(self, projection, c):
        pick = np.argmax(projection)
        pick_value = projection[pick]
        threshold = c * pick_value

        # Find left band
        left_pick_side = projection[0:pick]

        b0 = pick
        for index, intensity in reversed(list(enumerate(left_pick_side))):
            if intensity <= threshold:
                b0 = index
                break

        # Find right band
        right_pick_side = projection[pick + 1:projection.size + 1]

        b1 = pick
        for index, intensity in enumerate(right_pick_side):
            if intensity <= threshold:
                b1 = index
                break

        return b0, pick + b1

    def _find_y_bands(self, bands_count_limit=5):
        y_projection = np.sum(self.image, axis=1).tolist()
        before = y_projection = y_projection / np.max(y_projection)
        y_projection = signal.convolve(y_projection, self.mask, mode='same')

        utils.plot_histograms(before, y_projection, str(self.mask[4]))

        bands = []
        projection = np.copy(y_projection)
        for i in range(bands_count_limit):
            (y0, y1) = self._find_band(projection, c=self.y_c)
            bands.append((y0, y1))
            projection[y0:y1+1] = 0

        return bands

    def _find_x_bands(self, image, count=5, threshold=30):
        before = x_histogram = np.sum(image, axis=0).tolist()
        x_histogram = x_histogram / np.max(x_histogram)
        x_histogram = signal.convolve(x_histogram, self.mask, mode='same')

        # utils.plot_histograms(before, x_histogram, str(self.mask[4]))

        bands = []

        hist = np.copy(x_histogram)
        for i in range(count):
            (x0, x1) = self._find_band(hist, c=0.30)
            if x1-x0 >= threshold:
                bands.append((x0, x1))
                hist[x0:x1+1] = 0

        return bands

    def get_bands(self):
        bands = []

        for y0, y1 in self._find_y_bands():
            if y1-y0 <= 10:
                continue
            band = self.image[y0:y1, ...]
            x_bands = self._find_x_bands(band)
            [bands.append((y0, y1, x0, x1)) for x0, x1 in x_bands]

        return bands

    def last_step(self, bands):
        bonds = []

        for y0, y1, x0, x1 in bands:
            img = self.image[y0:y1, x0:x1]
            x_histogram = np.sum(img, axis=0).tolist()
            (nx0, nx1) = self._derivate(x_histogram)
            bonds.append((y0, y1, x0+nx0, x0+nx1))

        return bonds

    def _derivate(self, histogram, h=4, c=0.5):
        import math
        derivation = [((histogram[i] - histogram[i-h]) / h) for i in range(h, len(histogram))]
        center = math.ceil(len(derivation) / 2)
        # print("center", len(derivation), center)
        max_val = max(derivation)
        min_val = min(derivation)

        left = derivation[0:center]
        right = derivation[center + 1:]

        # Find upper bound
        # left = np.array([val if val >= c * max_val else 1 for val in np.flip(left, axis=0)])

        b0 = np.argmax(left)
        # b0 = (len(left) - b0)

        # Find lower bound
        # right = np.array([val if val <= c * min_val else 1 for val in right])
        b1 = np.argmin(right)

        return b0, center + b1


