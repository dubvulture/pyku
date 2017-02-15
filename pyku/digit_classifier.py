# coding=utf-8
import os

import cv2
import numpy as np

from .utils import FOLDER, DSIZE


class DigitClassifier(object):
    @staticmethod
    def _feature(image):
        image = cv2.resize(image, None, fx=DSIZE/28, fy=DSIZE/28,
                           interpolation=cv2.INTER_LINEAR)
        ret = image.astype(np.float32) / 255
        return ret.ravel()

    def __init__(self,
                 model=None,
                 train_set=FOLDER + '/Fnt/numeric_',
                 feature=_feature.__func__):
        self.feature = feature
        if model is not None:
            self.model = model
            self.train_set = None
        else:
            self.train_set = train_set
            self.model = self.create_model()

    def create_model(self):
        """
        Return a KNN classifier trained on self.train_set
        :return: KNN trained classifier
        """
        digits = []
        labels = []
        for n in range(1, 10):
            folder = self.train_set + str(n)
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
        return model

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
