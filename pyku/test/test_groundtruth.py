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


def test_groundtruth(standard=True):
    # Read all pictures
    folder = pyku.utils.FOLDER
    pics = sorted([os.path.join(folder, pic)
                   for pic in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, pic))])
    # Read groundtruth
    groundtruth = read_groundtruth()

    if standard:
        # Standard raw pixel data
        model = pyku.DigitClassifier()
    else:
        # Zoning data
        pyku.utils.DSIZE = 28.
        model = pyku.DigitClassifier(
            saved_model=pyku.utils.TRAIN_DATA + 'zoning_data.npz',
            feature=pyku.DigitClassifier._zoning)
    preds = []

    # How many images
    n = 52
    for i in range(n):
        if i == 19 or i == 1:
            # blurrer image, OUTLIER
            preds.append(None)
        else:
            pic = pics[i]
            im = pyku.Sudoku(pic, classifier=model)
            preds.append(im.extract(label_tries=3, debug=True))
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
                elif a!=0 and b!=0:
                    y_pred.append(a)
                    y_true.append(b)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.metrics import confusion_matrix
    logging.info(confusion_matrix(y_true, y_pred))
    logging.info('Recall: %f', recall_score(y_true, y_pred))
    logging.info('Precision: %f', precision_score(y_true, y_pred))
    logging.info('Accuracy: %f', accuracy_score(y_true, y_pred))
    logging.info('Wrong positions: %d', w_pos)

    for i in range(n):
        if preds[i] is None:
            logging.info('No grid found in %d.jpg', i+1)
        elif preds[i] != groundtruth[i]:
            logging.info('%d.jpg', i+1)
            logging.info(preds[i])
            logging.info(groundtruth[i])


if __name__ == "__main__":
    test_groundtruth(standard=False)
