import importlib
import logging

import click
import shutil

from statoil.models.model import Model
from statoil.utils import make_sure_path_exists, clean_folder, backup_files, pickle_load, json_load, json_dump, csv_dump, get_class
from statoil import cfg


@click.command()
@click.option('--exp', default="exp", help='Experiment folder name')
@click.option('--epochs', required=True, type=int, help='Nr of epochs to train the model')
@click.option('--short_run/--long_run', default=False, help='Run on short/long data')
def main(exp, epochs, short_run):
    exp_name = exp
    del exp

    # Clean
    exp_dir = cfg.EXPERIMENTS_DIR + exp_name + "/"
    backup_files([__file__, ], exp_dir)

    conf = json_load(exp_dir + "conf.json")
    traindev = pickle_load(cfg.DATASETS["traindev" if not short_run else "traindev_short"])

    # Model
    conf["model"]["__init__"]["epochs"] = epochs
    ModelClass = get_class(exp_dir + "used_model.py", conf["model"]["class"])
    model = ModelClass(conf, trained_on="traindev", **conf["model"]["__init__"])  # type: Model
    model.train_and_save(traindev, None, exp_dir)

if __name__ == '__main__':
    main()


