# coding=utf-8
import logging
import os
import sys

import numpy as np

import pyku

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def read_groundtruth():
    ret = []
    with open(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                'groundtruth.txt'), 'rb') as lines:
        for line in lines:
            ret.append(line[:-2])

    return np.array(ret)


def test_sudoku():
    # Read all pictures
    folder = pyku.utils.FOLDER
    pics = sorted([os.path.join(folder, pic)
                   for pic in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, pic))])
    # Read groundtruth
    groundtruth = read_groundtruth()

    # Train every time in case we change features
    model = pyku.DigitClassifier(train_folder=folder + 'Fnt/numeric_')
    preds = []

    # How many images
    n = 52
    for pic in pics[:n]:
        im = pyku.Sudoku(pic, classifier=model, debug=True)
        preds.append(im.extract(label_tries=3))
    preds = np.array(preds)

    res = np.equal(groundtruth[:n], preds)

    correct = np.size(res[res])
    nogrid = np.size(preds[np.equal(preds, None)])
    logging.info('Correct: %d', correct)
    logging.info('No grid: %d', nogrid)
    logging.info('Wrong digits: %d', (n - correct) - nogrid)

    w_pos = 0
    y_true = []
    y_pred = []
    for i in range(n):
        pred = preds[i]
        gt = groundtruth[i]
        if pred is not None:
            for j in range(81):
                a = 0 if pred[j] == ' ' else int(pred[j])
                b = 0 if gt[j] == ' ' else int(gt[j])
                # Wrong position or noise
                if (a == 0 and b != 0) or (a != 0 and b == 0):
                    w_pos += 1
                else:
                    y_pred.append(a)
                    y_true.append(b)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred)
    logging.info(report)


if __name__ == "__main__":
    test_sudoku()
