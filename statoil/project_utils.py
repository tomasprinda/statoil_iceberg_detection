import math

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth_p(dataset, eps=1e-4):
    for example in dataset:
        example["p"] = min(1 - eps, max(eps, example["p"]))


def dataset2arrays(dataset, return_labels=True, transfer_learning=False):
    xs, labels = [], []
    for example in dataset:
        if transfer_learning:
            xs.append(example["bands"])
        else:
            band_3 = example["band_1"] + example["band_2"]
            xs.append(np.stack((example["band_1"], example["band_2"], band_3), axis=2))
        if return_labels:
            labels.append(np.array([example["is_iceberg"], ], dtype=np.float32))

    xs = np.stack(xs, axis=0)
    if return_labels:
        labels = np.stack(labels, axis=0)
        return xs, labels
    return xs


def plot_examples(examples, type="band"):
    if type not in ["band", "band_hist"]:
        raise Exception("wrong type")

    row_len = 4
    n = len(examples) * 2
    n_rows = math.ceil(n / row_len)

    width = 20
    im_size = (width + 5) / row_len
    height = im_size * n_rows

    f, ax_arr = plt.subplots(n_rows, row_len, figsize=(width, height))
    if n_rows > 1 and row_len > 1:
        ax_arr = [ax for sublist in ax_arr for ax in sublist]

    for ax in ax_arr:
        if type == "band":
            ax.axis('off')
    for idx, example in enumerate(examples):
        title = "\n".join(["{}: {}".format(attr, example[attr]) for attr in ["id", "is_iceberg", "prediction", "p"] if attr in example])

        try:
            if type == "band_hist":
                ax_arr[2 * idx].hist(example["band_1"].ravel())
                ax_arr[2 * idx + 1].hist(example["band_2"].ravel())
            else:
                ax_arr[2 * idx].imshow(example["band_1"])
                ax_arr[2 * idx + 1].imshow(example["band_2"])
        except (FileNotFoundError, TypeError) as e:
            ax_arr[2 * idx].imshow(np.zeros((224, 224, 3)))
            ax_arr[2 * idx + 1].imshow(np.zeros((224, 224, 3)))

        if title is not None:
            color = "#222222"
            if "prediction" in example and example["is_iceberg"] != example["prediction"]:
                color = "#bf0000"
            ax_arr[2 * idx].set_title(title, color=color)
    return plt.show()


def plot_histograms(examples):
    row_len = 4
    n = len(examples) * 2
    n_rows = math.ceil(n / row_len)

    width = 20
    im_size = (width + 5) / row_len
    height = im_size * n_rows

    f, ax_arr = plt.subplots(n_rows, row_len, figsize=(width, height))
    if n_rows > 1 and row_len > 1:
        ax_arr = [ax for sublist in ax_arr for ax in sublist]

    for ax in ax_arr:
        ax.axis('off')
    for idx, example in enumerate(examples):
        title = "\n".join(["{}: {}".format(attr, example[attr]) for attr in ["id", "is_iceberg", "prediction", "p"] if attr in example])

        try:

            ax_arr[2 * idx].imshow(example["band_1"])
            ax_arr[2 * idx + 1].imshow(example["band_2"])
        except (FileNotFoundError, TypeError) as e:
            ax_arr[2 * idx].imshow(np.zeros((224, 224, 3)))
            ax_arr[2 * idx + 1].imshow(np.zeros((224, 224, 3)))

        if title is not None:
            color = "#222222"
            if "prediction" in example and example["is_iceberg"] != example["prediction"]:
                color = "#bf0000"
            ax_arr[2 * idx].set_title(title, color=color)
    return plt.show()
