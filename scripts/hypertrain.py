import importlib
import logging
import random

import click
import shutil

from copy import deepcopy

from statoil.models.model import Model
from statoil.utils import make_sure_path_exists, clean_folder, backup_files, pickle_load, json_load, json_dump, csv_dump
from statoil import cfg


@click.command()
@click.option('--conf', default=cfg.CONF_CNN_INCEPTION, help='Model configuration file')
@click.option('--exp', default="exp", help='Experiment folder name')
@click.option('--short_run/--long_run', default=False, help='Run on short/long data')
def main(conf, exp, short_run):
    conf_file, exp_name = conf, exp
    del conf, exp

    # Clean
    hypertuner_exp_dir = cfg.EXPERIMENTS_DIR + exp_name + "/"
    make_sure_path_exists(hypertuner_exp_dir)
    clean_folder(hypertuner_exp_dir)
    backup_files([__file__, ], hypertuner_exp_dir)

    hypertuner_conf = json_load(conf_file)
    json_dump(hypertuner_conf, hypertuner_exp_dir + "conf.json")
    trainset = pickle_load(cfg.DATASETS["train" if not short_run else "train_short"])
    devset = pickle_load(cfg.DATASETS["dev"])

    # Model
    module = importlib.import_module(hypertuner_conf["package"])
    shutil.copy(module.__file__, hypertuner_exp_dir + "used_model.py")
    ModelClass = getattr(module, hypertuner_conf["class"])

    csv_dump([["variant", "min_loss"]], hypertuner_exp_dir + "variants.csv")

    i = 0
    while True:
        i += 1
        exp_dir = cfg.EXPERIMENTS_DIR + "{}_{:03d}".format(exp_name, i) + "/"
        make_sure_path_exists(exp_dir)
        clean_folder(exp_dir)

        conf = pick_variant(hypertuner_conf)
        json_dump(conf, exp_dir + "conf.json")

        shutil.copy(module.__file__, exp_dir + "used_model.py")
        model = ModelClass(conf, trained_on="train")  # type: Model
        min_loss = model.train_and_save(trainset, devset, exp_dir)

        csv_dump([[i, min_loss]], hypertuner_exp_dir + "variants.csv", append=True)

        del model


def pick_variant(hypertuner_conf):
    conf = {}
    for key, val in hypertuner_conf.items():
        if isinstance(val, dict) and "hyperTuner" in val:
            val = random.choice(val["hyperTuner"])
        conf[key] = val

    return conf

if __name__ == '__main__':
    main()
