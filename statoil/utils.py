import csv
import errno
import importlib
import os
import pickle
import random
import shutil
import ujson

import numpy as np

from statoil import cfg


def prepare_exp_dir(exp_name, clean_dir):
    exp_dir = cfg.EXPERIMENTS_DIR + exp_name + "/"
    if clean_dir:
        make_sure_path_exists(exp_dir)
        clean_folder(exp_dir)
    backup_files([__file__, ], exp_dir)
    return exp_dir


def get_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, class_name)


def clean_folder(folder):
    import os
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def variable_summaries(var, name):
    import tensorflow as tf
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        if len(var.shape.as_list()) > 0:
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)

            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)


def backup_files(files, folder):
    for file in files:
        shutil.copy(file, folder)


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def make_sure_path_exists(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def shuffle_pairs(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def batches(l, batch_size, shuffle=False):
    """
    :param list|np.ndarray l:
    :param int batch_size:
    :param bool shuffle:
    """
    if shuffle:
        np.random.shuffle(l)
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]


def unbatch(batches_):
    for batch in batches_:
        for example in batch:
            yield example


def json_load(filename):
    with open(filename, "r") as f:
        return ujson.load(f)


def json_dump(d, filename):
    with open(filename, "w") as f:
        ujson.dump(d, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_dump(d, filename):
    with open(filename, "wb") as f:
        pickle.dump(d, f)


def csv_dump(rows, filename, append=False, delimiter=";"):
    open_param = 'a' if append else 'w'
    with open(filename, open_param) as csvfile:
        writer = csv.writer(csvfile, delimiter=str(delimiter), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)
        rows = [list(row) for row in rows]  # Like copy, but converts inner tuples to lists
        writer.writerows(rows)


def strip_extension(filename):
    arr = filename.split(".")
    if len(arr) == 1:
        return arr[0]
    return ".".join(arr[:-1])


def print_info(var, name="var", output=False):
    """
    Print variable information.
    :type var: any
    :type name: str
    :return:
    """
    if isinstance(var, np.ndarray):
        out = "{}: type:{}, shape:{}, dtype:{}, min:{}, max:{}".format(name, type(var), var.shape, var.dtype, np.min(var), np.max(var))
    elif isinstance(var, list) or isinstance(var, tuple):
        out = "{}: type:{}, len:{}, type[0]:{}".format(name, type(var), len(var), type(var[0]) if len(var) > 0 else "")
    else:
        out = "{}: val:{}, type:{}".format(name, var, type(var))
    if output:
        return out
    else:
        print(out)
