import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from statoil import cfg


class ExampleGenerator:
    def __init__(self):
        self.dataset = None

        self.band_1_med = None
        self.band_1_std = None
        self.band_1_visibility = None

        self.band_2_med = None
        self.band_2_std = None
        self.band_2_visibility = None

    def fit(self, dataset):
        self.dataset = dataset

        band_1 = np.stack([ex["band_1"] for ex in self.dataset])
        self.band_1_med = np.median(band_1, axis=(1, 2))
        self.band_1_std = np.std(band_1, axis=(1, 2))
        self.band_1_visibility = np.maximum(0, np.max(band_1, axis=(1, 2)) - 3 * self.band_1_std - self.band_1_med)

        band_2 = np.stack([ex["band_2"] for ex in self.dataset])
        self.band_2_med = np.median(band_2, axis=(1, 2))
        self.band_2_std = np.std(band_2, axis=(1, 2))
        self.band_2_visibility = np.maximum(0, np.max(band_2, axis=(1, 2)) - 3 * self.band_2_std - self.band_2_med)

    def generate(self, n):
        xs = []
        for x in np.linspace(-5, 5, cfg.IMG_SIZE):
            for y in np.linspace(-5, 5, cfg.IMG_SIZE):
                xs.append([x, y])

        for i in range(n):
            is_iceberg = np.random.randint(2)

            nr_iceboats = np.random.randint(2, 6) if is_iceberg else 1

            iceboat = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE))  # iceboat is part of iceberg or a boat
            half = int(cfg.IMG_SIZE / 2)
            for i in range(nr_iceboats):
                width = np.random.randint(1, 4)
                lenght = np.random.randint(2, 20)
                center_x, center_y = half + np.random.randint(0, 10, 2)
                degrees = np.random.randint(0, 90)
                iceboat[center_x - width:center_x + width, center_y - lenght:center_y + lenght] = 1
                iceboat = ndimage.rotate(iceboat, degrees, reshape=False, mode="mirror")
            iceboat = gaussian_filter(iceboat, sigma=2)
            iceboat /= np.max(iceboat)

            mu = np.random.choice(self.band_1_med)
            std = np.random.choice(self.band_1_std)
            visibility = np.random.choice(self.band_1_visibility)
            band_1 = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE)) + mu  # Base
            band_1 += np.random.normal(0, std, (cfg.IMG_SIZE, cfg.IMG_SIZE))  # Noise
            band_1 = gaussian_filter(band_1, sigma=0.8)
            band_1 += iceboat * visibility  # object

            mu = np.random.choice(self.band_2_med)
            std = np.random.choice(self.band_2_std)
            visibility = np.random.choice(self.band_2_visibility)
            band_2 = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE)) + mu  # Base
            band_2 += np.random.normal(0, std, (cfg.IMG_SIZE, cfg.IMG_SIZE))  # Noise
            band_2 = gaussian_filter(band_2, sigma=0.8)
            band_2 += iceboat * visibility  # object

            example = {
                "id": "0",
                "synthetic": True,
                "synthetic_method": "generated",
                "is_iceberg": is_iceberg,
                "inc_angle": 0.,
                "band_1": band_1,
                "band_2": band_2,
            }

            yield example
