import numpy as np
from scipy import signal


class BindsFinder:

    def __init__(self, image):
        self.image = np.array(image / np.max(image))

    def _find_band(self, histogram, c=0.55):
        pick = np.argmax(histogram)
        left = histogram[0:pick]
        right = histogram[pick+1:histogram.size+1]

        pick_value = histogram[pick]

        # Find upper bound
        left = np.array([val if val <= c * pick_value else 0 for val in np.flip(left, axis=0)])
        if left.size == 0:
            return 0, 0

        b0 = np.argmax(left)
        b0 = (left.size - b0)

        # Find lower bound
        right = np.array([val if val <= c * pick_value else 1 for val in right])
        b1 = np.argmin(right)

        return b0, pick+b1

    def _find_y_bands(self, count=5, threshold=10):
        y_histogram = np.sum(self.image, axis=1).tolist()
        y_histogram = signal.convolve(y_histogram, [1, 1, 1, 0, 0, 0, 1, 1, 1], mode='same')
        bands = []

        hist = np.copy(y_histogram)
        for i in range(count):
            (y0, y1) = self._find_band(hist)
            if y1-y0 >= threshold:
                bands.append((y0, y1))
                hist[y0:y1] = 0

        return bands

    def _find_x_bands(self, image, count=5, threshold=30):
        x_histogram = np.sum(image, axis=0).tolist()
        x_histogram = signal.convolve(x_histogram, [1, 1, 1, 0, 0, 0, 1, 1, 1], mode='same')

        bands = []

        hist = np.copy(x_histogram)
        for i in range(count):
            (x0, x1) = self._find_band(hist, c=0.42)
            if x1-x0 >= threshold:
                bands.append((x0, x1))
                hist[x0:x1] = 0

        return bands

    def get_bands(self):
        bands = []
        for y0, y1 in self._find_y_bands():
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
        print("center", len(derivation), center)
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


