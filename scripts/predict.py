import importlib
import importlib.util
import click

from statoil import cfg
from statoil.models.model import Model
from statoil.project_utils import smooth_p
from statoil.utils import backup_files, pickle_load, csv_dump, json_load, json_dump, pickle_dump, get_class


@click.command()
@click.option('--exp', default="exp", help='Experiment folder name with a model.')
@click.option('--dataset', default="test", help='{train|dev|traindev|test}[_short]')
@click.option('--traindev_model/--train_model', default=False, help='What model to use')
def main(exp, dataset, traindev_model):
    exp_name, dataset_name, use_traindev_model = exp, dataset, traindev_model
    del exp
    del dataset
    del traindev_model

    # Clean
    exp_dir = cfg.EXPERIMENTS_DIR + exp_name + "/"
    backup_files([__file__], exp_dir)

    # Data
    dataset = pickle_load(cfg.DATASETS[dataset_name])
    conf = json_load(exp_dir + "conf.json")

    # Model
    trained_on = "traindev" if use_traindev_model else "train"
    ModelClass = get_class(exp_dir + "used_model.py", conf["class"])
    model = ModelClass(conf, trained_on=trained_on)  # type: Model
    model.predict(dataset, exp_dir)
    # smooth_p(dataset)

    # Store
    pickle_dump(dataset, exp_dir + dataset_name + "_predictions.pkl")
    submission_dump(dataset, exp_dir + "{}Model_{}Data_submission.csv".format(trained_on, dataset_name))


def submission_dump(dataset, filename):
    header = ["id", "is_iceberg"]
    rows = [[row["id"], "{:.6f}".format(row["p"])] for row in dataset]
    csv_dump([header] + rows, filename, delimiter=",")


if __name__ == '__main__':
    main()
