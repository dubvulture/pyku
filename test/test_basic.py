# coding=utf-8
import logging
import os
import sys
import unittest

import numpy as np

import pyku


class TestBasic(unittest.TestCase):

    @staticmethod
    def _read_groundtruth():
        ret = []
        with open('./groundtruth.txt', 'rb') as lines:
            for line in lines:
                ret.append(line[:-2])

        return np.array(ret)

    @classmethod
    def setUpClass(cls):
        cls.folder = pyku.utils.FOLDER
        cls.groundtruth = TestBasic._read_groundtruth()
        logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    def test_classifier(self):
        model1 = pyku.DigitClassifier()
        model2 = pyku.DigitClassifier(train_folder=self.folder + 'Fnt/numeric_')
        self.assertTrue((model1.train_set==model2.train_set).all())
        self.assertTrue((model1.train_labels==model2.train_labels).all())

    def test_groundtruth(self):
        pics = sorted([os.path.join(self.folder, pic)
                       for pic in os.listdir(self.folder)
                       if os.path.isfile(os.path.join(self.folder, pic))])
        model = pyku.DigitClassifier()
        preds = []
        n = 52
        for pic in pics[:n]:
            im = pyku.Sudoku(pic, classifier=model, debug=True)
            preds.append(im.extract(label_tries=3))
        preds = np.array(preds)

        res = self.groundtruth[:n] == preds

        correct = np.size(res[res])
        nogrid = np.size(preds[preds == None])
        logging.critical('Correct: %d', correct)
        logging.critical('No grid: %d', nogrid)
        logging.critical('Wrong digits: %d', (n - correct) - nogrid)
        self.assertGreater(np.size(res[res]), 44)


if __name__ == '__main__':
    unittest.main()
