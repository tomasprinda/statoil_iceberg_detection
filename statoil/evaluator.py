import random
import sklearn.metrics as metrics

from statoil.project_utils import plot_examples


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_confusion_matrix(self):
        y_true, y_pred = zip(*[(example["is_iceberg"], example["prediction"]) for example in self.dataset])
        cm = metrics.confusion_matrix(y_true, y_pred)
        return cm

    def get_log_loss(self):
        y_true, y_pred = zip(*[(example["is_iceberg"], example["p"]) for example in self.dataset])
        log_loss = metrics.log_loss(y_true, y_pred)
        return log_loss

    def plot_confusion_matrix(self):
        cm = self.get_confusion_matrix()
        print("True")
        print(" " * 5 + "Predict")
        print(cm)

    def plot_predictions(self, is_iceberg, prediction):
        examples = []
        for example in self.dataset:
            if (example["is_iceberg"] == is_iceberg or is_iceberg is None) and \
                    (example["prediction"] == prediction or prediction is None):
                examples.append(example)
        random.shuffle(examples)
        plot_examples(examples[:10])
