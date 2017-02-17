# coding=utf-8
import logging
import os
import sys

import pyku

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def test_performance(standard=True):
    folder = pyku.utils.FOLDER
    pics = sorted([os.path.join(folder, pic)
                   for pic in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, pic))])

    if standard:
        # Standard raw pixel data
        model = pyku.DigitClassifier()
    else:
        # Zoning data
        pyku.utils.DSIZE = 28.
        model = pyku.DigitClassifier(
            saved_model=pyku.utils.TRAIN_DATA+'zoning_data.npz',
            feature=pyku.DigitClassifier._zoning)

    for pic in pics[:52]:
        a = pyku.Sudoku(pic, classifier=model)
        a.extract()

    return None



if __name__ == '__main__':
    test_performance(standard=False)