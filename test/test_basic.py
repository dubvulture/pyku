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

    def test_groundtruth(self):
        pics = sorted([os.path.join(self.folder, pic)
                       for pic in os.listdir(self.folder)
                       if os.path.isfile(os.path.join(self.folder, pic))])
        model = pyku.DigitClassifier()
        preds = []
        n = 52
        for pic in pics[:n]:
            grabber = pyku.Sudoku(pic, classifier=model, debug=True)
            preds.append(grabber.grab(label_tries=3))
        preds = np.array(preds)

        res = self.groundtruth[:n] == preds

        with open('./res1.txt', 'wb') as f:
            for i in range(0, res.size):
                if res[i]:
                    f.write(str(i) + '\n')

        nogrid = np.size(preds[preds == None])
        logging.critical('No grid found: %d', nogrid)
        logging.critical('Wrong digits: %d', (n - np.size(res[res])) - nogrid)
        self.assertGreater(np.size(res[res]), 44, msg=':(')


if __name__ == '__main__':
    unittest.main()
