# coding=utf-8
import os

import cv2
import numpy as np

from .utils import DSIZE, KNN_DATA


class DigitClassifier(object):
    @staticmethod
    def _feature(image):
        image = cv2.resize(image, None, fx=DSIZE/28, fy=DSIZE/28,
                           interpolation=cv2.INTER_LINEAR)
        ret = image.astype(np.float32) / 255
        return ret.ravel()

    @staticmethod
    def _save_model():
        knn = DigitClassifier()
        np.savez(KNN_DATA, train_set=knn.train_set, train_labels=knn.train_labels)

    def __init__(self,
                 saved_model=None,
                 train_folder=None,
                 feature=_feature.__func__):
        self.feature = feature
        if train_folder is not None:
            self.train_set, self.train_labels, self.model = \
                self.create_model(train_folder)
        else:
            saved_model = KNN_DATA if saved_model is None else saved_model
            self.model = cv2.KNearest()
            with np.load(saved_model) as data:
                self.train_set = data['train_set']
                self.train_labels = data['train_labels']
                self.model.train(self.train_set, self.train_labels)


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
        model = cv2.KNearest()
        model.train(digits, labels)
        return digits, labels, model

    def classify(self, image):
        """
        Given a 28x28 image, returns an array representing the 2 highest
        probable prediction
        :param image:
        :return: array of 2 highest prob-digit tuples
        """
        res = self.model.find_nearest(np.array([self.feature(image)]), k=11)
        hist = np.histogram(res[2], bins=9, range=(1, 10), normed=True)[0]
        zipped = sorted(zip(hist, np.arange(1, 10)), reverse=True)
        return np.array(zipped[:2])
