# coding=utf-8
import os

import cv2
import numpy as np

from .utils import DSIZE, TRAIN_DATA


class DigitClassifier(object):
    @staticmethod
    def _feature(image):
        """
        It's faster but still accurate enough with DSIZE = 14.
        ~0.9983 precision and recall
        :param image:
        :return: raw pixels as feature vector
        """
        image = cv2.resize(image, None, fx=DSIZE/28, fy=DSIZE/28,
                           interpolation=cv2.INTER_LINEAR)
        ret = image.astype(np.float32) / 255
        return ret.ravel()

    @staticmethod
    def _zoning(image):
        """
        It works better with DSIZE = 28
        ~0.9967 precision and recall
        :param image:
        :return: #pixels/area ratio of each zone (7x7) as feature vector
        """
        zones = []
        for i in range(0, 28, 7):
            for j in range(0, 28, 7):
                roi = image[i:i+7, j:j+7]
                val = (np.sum(roi)/255) / 49.
                zones.append(val)
        return np.array(zones, np.float32)

    def __init__(self,
                 saved_model=None,
                 train_folder=None,
                 feature=_feature.__func__):
        """
        :param saved_model: optional saved train set and labels as .npz
        :param train_folder: optional custom train data to process
        :param feature: feature function - compatible with saved_model
        """
        self.feature = feature
        if train_folder is not None:
            self.train_set, self.train_labels, self.model = \
                self.create_model(train_folder)
        else:
            if cv2.__version__[0] == '2':
                self.model = cv2.KNearest()
            else:
                self.model = cv2.ml.KNearest_create()
            if saved_model is None:
                saved_model = TRAIN_DATA+'raw_pixel_data.npz'
            with np.load(saved_model) as data:
                self.train_set = data['train_set']
                self.train_labels = data['train_labels']
                if cv2.__version__[0] == '2':
                    self.model.train(self.train_set, self.train_labels)
                else:
                    self.model.train(self.train_set, cv2.ml.ROW_SAMPLE,
                                     self.train_labels)

    def create_model(self, train_folder):
        """
        Return the training set, its labels and the trained model
        :param train_folder: folder where to retrieve data
        :return: (train_set, train_labels, trained_model)
        """
        digits = []
        labels = []
        for n in range(1, 10):
            folder = train_folder + str(n)
            samples = [pic for pic in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, pic))]

            for sample in samples:
                image = cv2.imread(os.path.join(folder, sample))
                # Expecting black on white
                image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, image = cv2.threshold(image, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                feat = self.feature(image)
                digits.append(feat)
                labels.append(n)

        digits = np.array(digits, np.float32)
        labels = np.array(labels, np.float32)
        if cv2.__version__[0] == '2':
            model = cv2.KNearest()
            model.train(digits, labels)
        else:
            model = cv2.ml.KNearest_create()
            model.train(digits, cv2.ml.ROW_SAMPLE, labels)
        return digits, labels, model

    def classify(self, image):
        """
        Given a 28x28 image, returns an array representing the 2 highest
        probable prediction
        :param image:
        :return: array of 2 highest prob-digit tuples
        """
        if cv2.__version__[0] == '2':
            res = self.model.find_nearest(np.array([self.feature(image)]), k=11)
        else:
            res = self.model.findNearest(np.array([self.feature(image)]), k=11)
        hist = np.histogram(res[2], bins=9, range=(1, 10), normed=True)[0]
        zipped = sorted(zip(hist, np.arange(1, 10)), reverse=True)
        return np.array(zipped[:2])

    def save_training(self, filename):
        """
        Save traning set and labels of current model
        :param filename: filename of new data.npz, it will be saved in 'train/'
        """
        np.savez(os.path.join(TRAIN_DATA, filename),
                 train_set=self.train_set,
                 train_labels=self.train_labels)