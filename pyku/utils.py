# coding=utf-8
import os

import cv2
import numpy as np

__all__ = [
    'FOLDER', 'CROSS', 'ONES', 'WSIZE', 'SSIZE', 'DSIZE',
    'CENTROIDS', 'KNN_DATA', 'min_side', 'diagonal', 'compute_angle',
    'compute_line', 'morph']

FOLDER = '/home/manuel/Dropbox/Università/3° A.A/1° Sem/' \
         'Elaborazione delle Immagini/immagini/'

CROSS = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], np.uint8)


def ONES(n):
    return np.ones((n, n), np.uint8)


WSIZE = [3, 5, 7]

# Sudoku Size (when extracting)
SSIZE = 297
assert((SSIZE % 9) == 0)

# Digit Size (for classifier)
# Higher values increase running time of find_nearest but decrease accuracy
# 28 => 48% of running time
# 14 => 21% of running time
# (but only with NEAREST to 28 -> LINEAR to DSIZE when classifying test digits)
# (resizing directly extracted digits to DSIZE greatly lowers accuracy)
DSIZE = 14.0
assert(DSIZE <= 28)


def _gen_centroids():
    a = np.arange(SSIZE/18, SSIZE, SSIZE/9)
    x, y = np.meshgrid(a, a)
    return np.dstack((y, x)).reshape((81, 2))

CENTROIDS = _gen_centroids()

KNN_DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        '../train/knn_data.npz')


def min_side(_, pos):
    """
    Given an object pixels' positions, return the minimum side length of its
    bounding box
    :param _: pixel values (unused)
    :param pos: pixel position (1-D)
    :return: minimum bounding box side length
    """
    xs = np.array([i / SSIZE for i in pos])
    ys = np.array([i % SSIZE for i in pos])
    minx = np.amin(xs)
    miny = np.amin(ys)
    maxx = np.amax(xs)
    maxy = np.amax(ys)
    ct1 = compute_line(np.array([minx, miny]), np.array([minx, maxy]))
    ct2 = compute_line(np.array([minx, miny]), np.array([maxx, miny]))
    return min(ct1, ct2)


def diagonal(_, pos):
    """
    Given an object pixels' positions, return the diagonal length of its
    bound box
    :param _: pixel values (unused)
    :param pos: pixel position (1-D)
    :return: diagonal of bounding box
    """
    xs = np.array([i / SSIZE for i in pos])
    ys = np.array([i % SSIZE for i in pos])
    minx = np.amin(xs)
    miny = np.amin(ys)
    maxx = np.amax(xs)
    maxy = np.amax(ys)
    return compute_line(np.array([minx, miny]), np.array([maxx, maxy]))


def compute_angle(pt0, pt1, pt2):
    """
    Given 3 points, compute the cosine of the angle from pt0
    :type pt0: numpy.array
    :type pt1: numpy.array
    :type pt2: numpy.array
    :return: cosine of angle
    """
    a = pt0 - pt1
    b = pt0 - pt2
    return (np.sum(a * b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_line(pt0, pt1):
    """
    Given 2 points, compute their distance
    :type pt0: numpy.array
    :type pt1: numpy.array
    :return: distance
    """
    return np.linalg.norm(pt0 - pt1)


def morph(roi):
    ratio = min(28. / np.size(roi, 0), 28. / np.size(roi, 1))
    roi = cv2.resize(roi, None, fx=ratio, fy=ratio,
                     interpolation=cv2.INTER_NEAREST)
    dx = 28 - np.size(roi, 1)
    dy = 28 - np.size(roi, 0)
    px = ((int(dx / 2.)), int(np.ceil(dx / 2.)))
    py = ((int(dy / 2.)), int(np.ceil(dy / 2.)))
    squared = np.pad(roi, (py, px), 'constant', constant_values=0)
    return squared
