import pickle
import random
from scipy import ndimage

import numpy as np
from scipy.misc import imresize

from statoil import cfg
from statoil.example_generator import ExampleGenerator
from statoil.utils import json_load, print_info, pickle_dump, make_sure_path_exists, clean_folder, backup_files
import copy

RATIO_TRAIN = 0.8
SHORT_SIZE = 100


def main():
    # Clean
    make_sure_path_exists(cfg.DATA_FOLDER)
    clean_folder(cfg.DATA_FOLDER)
    backup_files([__file__], cfg.DATA_FOLDER)

    # Trainset
    traindev = json_load(cfg.RAW_TRAINSET)
    prepare_examples(traindev)
    random.shuffle(traindev)
    split = int(len(traindev) * RATIO_TRAIN)
    train_thin, dev_thin = copy.deepcopy(traindev[:split]), copy.deepcopy(traindev[split:])
    train = expand_dataset(train_thin)
    traindev = expand_dataset(traindev)
    dev = expand_dataset(dev_thin)

    # Testset
    test = json_load(cfg.RAW_TESTSET)
    prepare_examples(test)

    # Genset
    # gen = ExampleGenerator()
    # gen.fit(train_thin)
    # gentrain = gen.generate(40000)
    # gendev = gen.generate(5000)

    # Dump
    datasets = {
        "traindev": traindev,
        "train": train,
        "dev": dev,
        "test": test,
    }

    for name, dataset in datasets.items():
        print("Processing {}".format(name))
        dataset = list(dataset)

        # dataset = transform_to_tl(dataset)

        pickle_dump(dataset, cfg.DATASETS[name])

        name_short = name + "_short"
        pickle_dump(dataset[:SHORT_SIZE], cfg.DATASETS[name_short])

        del dataset


def prepare_examples(dataset):
    for example in dataset:
        example["band_1"] = prepare_image(example["band_1"])
        example["band_2"] = prepare_image(example["band_2"])


def prepare_image(img_list):
    img_arr = np.array(img_list, dtype=np.float32)
    img_arr = np.reshape(img_arr, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    return img_arr


def expand_dataset(dataset):
    for example in dataset:
        example["synthetic"] = 0
        example["synthetic_method"] = None
        yield example

        example_flip = flip_example(example, hflip=True, vflip=False)
        yield example_flip

        for degrees in range(90, 360, 90):
            yield rotate_example(example, degrees)
            yield rotate_example(example_flip, degrees)


def flip_example(example, hflip, vflip):
    synth_example = copy.deepcopy(example)
    synth_example["synthetic"] = 1
    synth_example["synthetic_method"] = ""

    if hflip:
        synth_example["synthetic_method"] += "Hflip"
        synth_example["band_1"] = np.flip(synth_example["band_1"], axis=1)
        synth_example["band_2"] = np.flip(synth_example["band_2"], axis=1)

    if vflip:
        synth_example["synthetic_method"] += "Vflip"
        synth_example["band_1"] = np.flip(synth_example["band_1"], axis=0)
        synth_example["band_2"] = np.flip(synth_example["band_2"], axis=0)

    synth_example["id"] += "_" + synth_example["synthetic_method"]
    return synth_example


def rotate_example(example, degrees):
    synth_example = copy.deepcopy(example)
    synth_example["synthetic"] = 1
    synth_example["synthetic_method"] = "rotate{}".format(degrees)

    synth_example["band_1"] = ndimage.rotate(synth_example["band_1"], degrees, reshape=False, mode="mirror")
    synth_example["band_2"] = ndimage.rotate(synth_example["band_2"], degrees, reshape=False, mode="mirror")

    synth_example["id"] += "_" + synth_example["synthetic_method"]
    return synth_example


def transform_to_tl(dataset):
    for example in dataset:
        example["bands"] = np.stack(
            [example["band_1"], example["band_2"], example["band_1"] - example["band_2"]],
            axis=2
        )
        example["bands"] = imresize(example["bands"], (cfg.IMG_SIZE_TL, cfg.IMG_SIZE_TL))
        del example["band_1"]
        del example["band_2"]
    return dataset


if __name__ == '__main__':
    main()
