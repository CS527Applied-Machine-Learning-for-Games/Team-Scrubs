import numpy as np
import math
# from itertools import izip

from keras import backend as K

from utils.image_augmentation import ImageDataGenerator


class InputArrays(object):

    def __init__(self,
                 image_train, label_train, image_valid, label_valid,
                 image_test=None, label_test=None, class_mode=None
                 ):
        self.image_train = image_train
        self.label_train = label_train
        self.image_valid = image_valid
        self.label_valid = label_valid
        self.image_test = image_test if image_test is not None else image_valid
        self.label_test = label_test if label_test is not None else label_valid

        self.image_train = np.asarray(self.image_train, dtype=K.floatx()) if self.image_train is not None else None
        self.label_train = np.asarray(self.label_train, dtype=K.floatx()) if self.label_train is not None else None
        self.image_valid = np.asarray(self.image_valid, dtype=K.floatx()) if self.image_valid is not None else None
        self.label_valid = np.asarray(self.label_valid, dtype=K.floatx()) if self.label_valid is not None else None
        self.image_test = np.asarray(self.image_test, dtype=K.floatx()) if self.image_test is not None else None
        self.label_test = np.asarray(self.label_test, dtype=K.floatx()) if self.label_test is not None else None

        self.class_mode = class_mode

    def get_train_flow(self, batch_size, aug_args, shuffle):
        raise NotImplementedError

    def get_valid_flow(self, batch_size, shuffle):
        return ImageDataGenerator().flow(self.image_valid, self.label_valid, batch_size=batch_size, shuffle=shuffle)

    def get_test_flow(self, batch_size):
        return ImageDataGenerator().flow(self.image_test, self.label_test, batch_size=batch_size, shuffle=False)

    def get_train_num_batches(self, batch_size):
        return int(math.ceil(len(self.image_train) / float(batch_size)))

    def get_valid_num_batches(self, batch_size):
        return int(math.ceil(len(self.image_valid) / float(batch_size)))

    def get_test_num_batches(self, batch_size):
        return int(math.ceil(len(self.image_test) / float(batch_size)))

    def get_num_classes(self):
        return int(np.max(self.label_train)) + 1

    def get_class_weights(self, class_weights_exp):
        class_frequencies = np.array([np.sum(self.label_train == f) for f in range(self.get_num_classes())])
        class_weights = class_frequencies.sum() / (class_frequencies.astype(np.float32))
        return class_weights ** class_weights_exp

    # def get_class_weights_multilabel(self, class_weights_exp, use_weight_balance=True):
    #     """The weights between the positive and negative classes of EACH label."""
    #     class_frequencies = np.array([np.sum(self.label_train == f) for f in range(self.get_num_classes())])
    #     class_frequencies = np.asarray(class_frequencies, dtype=np.float)
    #
    #     pos_weights = (1 - (class_frequencies / len(self.label_train))) ** class_weights_exp
    #     neg_weights = (1 - pos_weights) ** class_weights_exp
    #
    #     if use_weight_balance:
    #         reciprocal = np.reciprocal(class_frequencies)
    #         factor = len(reciprocal) * reciprocal / reciprocal.sum()
    #         pos_weights *= factor
    #         neg_weights *= factor
    #
    #     return pos_weights, neg_weights

    def get_class_weights_multilabel(self, class_weights_exp):

        class_frequencies = np.array([np.sum(self.label_train == f) for f in range(self.get_num_classes())])

        pos_frequencies = np.asarray(class_frequencies, dtype=np.float)
        neg_frequencies = len(self.label_train) - pos_frequencies

        total_freq = pos_frequencies.sum() + neg_frequencies.sum()

        pos_weights = (total_freq / pos_frequencies) ** class_weights_exp
        neg_weights = (total_freq / neg_frequencies) ** class_weights_exp

        return pos_weights, neg_weights

    def get_class_mode(self):
        return self.class_mode


class InputClassificationArrays(InputArrays):

    def get_train_flow(self, batch_size, aug_args, shuffle):
        if aug_args is None:
            aug_args = {}
        return ImageDataGenerator(**aug_args).flow(
            self.image_train, self.label_train, batch_size=batch_size, shuffle=shuffle)


class InputSegmentationArrays(InputArrays):

    def get_train_flow(self, batch_size, aug_args, shuffle):
        if aug_args is None:
            aug_args = {}
        generator = ImageDataGenerator(**aug_args)
        seed = np.random.randint(1000)
        image_gen = generator.flow(self.image_train, batch_size=batch_size, shuffle=shuffle, seed=seed)
        label_gen = generator.flow(self.label_train, batch_size=batch_size, shuffle=shuffle, seed=seed)
        return zip(image_gen, label_gen)


class InputLists(object):

    def __init__(self, directory, list_train, list_valid, list_test=None, class_mode='binary'):
        self.directory = directory
        self.list_train = list_train
        self.list_valid = list_valid
        self.list_test = list_test if list_test is not None else list_valid
        self.class_mode = class_mode

        self.train = []
        self.valid = []
        self.test = []
        if self.list_train is not None:
            with open(self.list_train) as f:
                self.train = f.read().splitlines()
        if self.list_valid is not None:
            with open(self.list_valid) as f:
                self.valid = f.read().splitlines()
        if self.list_test is not None:
            with open(self.list_test) as f:
                self.test = f.read().splitlines()

        # For num_classes and class weights computations
        self.label_train = []
        if self.train:
            for l in self.train:
                _, label = l.split()
                if self.class_mode == 'binary':  # i.e. 011010
                    binary = np.array([int(i) for i in label], dtype=np.uint8)
                    self.label_train.append(binary)
                else:
                    self.label_train.append(int(label))
            self.label_train = np.array(self.label_train)

    def get_train_flow(self, batch_size, aug_args, shuffle, target_image_size, num_input_channels=1):
        if aug_args is None:
            aug_args = {}
        return ImageDataGenerator(**aug_args).flow_from_list(
            self.list_train, directory=self.directory, target_size=target_image_size,
            num_input_channels=num_input_channels, class_mode=self.class_mode, batch_size=batch_size, shuffle=shuffle)

    def get_valid_flow(self, batch_size, shuffle, target_image_size, num_input_channels=1):
        return ImageDataGenerator().flow_from_list(
            self.list_valid, directory=self.directory, target_size=target_image_size,
            num_input_channels=num_input_channels, class_mode=self.class_mode, batch_size=batch_size, shuffle=shuffle)

    def get_test_flow(self, batch_size, target_image_size, num_input_channels=1):
        return ImageDataGenerator().flow_from_list(
            self.list_test, directory=self.directory, target_size=target_image_size,
            num_input_channels=num_input_channels, class_mode=self.class_mode, batch_size=batch_size, shuffle=False)

    def get_train_num_batches(self, batch_size):
        return int(math.ceil(len(self.train) / float(batch_size)))

    def get_valid_num_batches(self, batch_size):
        return int(math.ceil(len(self.valid) / float(batch_size)))

    def get_test_num_batches(self, batch_size):
        return int(math.ceil(len(self.test) / float(batch_size)))

    def get_num_classes(self):
        if self.class_mode == 'binary':
            return self.label_train.shape[1]
        else:
            return np.max(self.label_train) + 1

    def get_class_weights(self, class_weights_exp):
        if self.class_mode == 'binary':
            class_frequencies = self.label_train.sum(0)
        else:
            num_classes = self.get_num_classes()
            class_frequencies = np.array([np.sum(self.label_train == l) for l in range(num_classes)])
        class_weights = class_frequencies.sum() / class_frequencies.astype(np.float32)
        return class_weights ** class_weights_exp

    # def get_class_weights_multilabel(self, class_weights_exp, use_weight_balance=True):
    #     """The weights between the positive and negative classes of EACH label."""
    #
    #     if self.class_mode == 'binary':
    #         class_frequencies = self.label_train.sum(0)
    #     else:
    #         num_classes = self.get_num_classes()
    #         class_frequencies = np.array([np.sum(self.label_train == l) for l in range(num_classes)])
    #     class_frequencies = np.asarray(class_frequencies, dtype=np.float)
    #
    #     pos_weights = (1 - (class_frequencies / len(self.label_train))) ** class_weights_exp
    #     neg_weights = (1 - pos_weights) ** class_weights_exp
    #
    #     if use_weight_balance:
    #         reciprocal = np.reciprocal(class_frequencies)
    #         factor = len(reciprocal) * reciprocal / reciprocal.sum()
    #         pos_weights *= factor
    #         neg_weights *= factor
    #
    #     return pos_weights, neg_weights

    def get_class_weights_multilabel(self, class_weights_exp):

        if self.class_mode == 'binary':
            class_frequencies = self.label_train.sum(0)
        else:
            num_classes = self.get_num_classes()
            class_frequencies = np.array([np.sum(self.label_train == l) for l in range(num_classes)])

        pos_frequencies = np.asarray(class_frequencies, dtype=np.float)
        neg_frequencies = len(self.label_train) - pos_frequencies

        total_freq = pos_frequencies.sum() + neg_frequencies.sum()

        pos_weights = (total_freq / pos_frequencies) ** class_weights_exp
        neg_weights = (total_freq / neg_frequencies) ** class_weights_exp

        return pos_weights, neg_weights

    def get_class_mode(self):
        return self.class_mode
