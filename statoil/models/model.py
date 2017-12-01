class Model:
    def __init__(self, conf, trained_on):
        self.conf = conf
        self.trained_on = trained_on
        pass

    def pretrain_and_save(self, trainset, exp_dir):
        raise NotImplemented()

    def train_and_save(self, trainset, devset, save_dir=None):
        raise NotImplemented()

    def predict(self, dataset, save_dir):
        raise NotImplemented()



