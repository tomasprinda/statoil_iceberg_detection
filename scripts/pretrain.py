import importlib
import logging

import click
import shutil

from statoil.models.model import Model
from statoil.utils import make_sure_path_exists, clean_folder, backup_files, pickle_load, json_load, json_dump, csv_dump
from statoil import cfg


@click.command()
@click.option('--conf', default=cfg.CONF_CNN_PRETRAIN, help='Model configuration file')
@click.option('--exp', default="exp", help='Experiment folder name')
@click.option('--short_run/--long_run', default=False, help='Run on short/long data')
def main(conf, exp, short_run):
    conf_file, exp_name = conf, exp
    del conf, exp

    # Clean
    exp_dir = cfg.EXPERIMENTS_DIR + exp_name + "/"
    make_sure_path_exists(exp_dir)
    clean_folder(exp_dir)
    backup_files([__file__, ], exp_dir)

    conf = json_load(conf_file)
    json_dump(conf, exp_dir + "conf.json")
    trainset = pickle_load(cfg.DATASETS["train" if not short_run else "train_short"])

    # Model
    module = importlib.import_module(conf["model"]["package"])
    shutil.copy(module.__file__, exp_dir+"used_model.py")
    ModelClass = getattr(module, conf["model"]["class"])
    model = ModelClass(conf, trained_on="gentrain", **conf["model"]["__init__"])  # type: Model
    model.pretrain_and_save(trainset, exp_dir)

if __name__ == '__main__':
    main()
