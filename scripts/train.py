import importlib
import logging

import click
import shutil

from statoil.models.model import Model
from statoil.utils import make_sure_path_exists, clean_folder, backup_files, pickle_load, json_load, json_dump, csv_dump, prepare_exp_dir
from statoil import cfg


@click.command()
@click.option('--conf', default=cfg.CONF_CNN_INCEPTION, help='Model configuration file')
@click.option('--exp', default="exp", help='Experiment folder name')
@click.option('--short_run/--long_run', default=False, help='Run on short/long data')
def main(conf, exp, short_run):
    conf_file, exp_name = conf, exp
    del conf, exp

    # Clean
    exp_dir = prepare_exp_dir(exp_name, clean_dir=True)

    conf = json_load(conf_file)
    json_dump(conf, exp_dir + "conf.json")
    trainset = pickle_load(cfg.DATASETS["train" if not short_run else "train_short"])
    devset = pickle_load(cfg.DATASETS["dev"])

    # Model
    module = importlib.import_module(conf["package"])
    shutil.copy(module.__file__, exp_dir+"used_model.py")
    ModelClass = getattr(module, conf["class"])
    model = ModelClass(conf, trained_on="train")  # type: Model
    model.train_and_save(trainset, devset, exp_dir)

if __name__ == '__main__':
    main()
