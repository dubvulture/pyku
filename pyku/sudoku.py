# coding=utf-8
import logging
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as extract_feature
from scipy.ndimage import find_objects
from scipy.spatial.distance import cdist

from .digit_classifier import DigitClassifier
from .utils import *

logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)


class Sudoku(object):

    def __init__(self, filename, classifier=None,
                 perspective=False, debug=False):
        """
        :param filename: image with sudoku
        :param classifier: digit classifier
        :param perspective: detect sudoku higly distorted by perspective or not,
            enabling it just deactivate sides length check
        :param debug: print/save debug messages/images
        """
        self.filename = os.path.basename(filename)
        image = cv2.imread(filename)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if classifier is None:
            self.classifier = DigitClassifier()
        else:
            self.classifier = classifier
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        self.perspective = perspective
        self.counter = 0
        self.step = -1

    def extract(self, label_tries=4, multiple=False):
        """
        Tries to extract a sudoku from a given image
        :param label_tries: number of times it tries to find a grid in the image
        :param multiple: indicates whether there is one or more sudoku to grab
        :return: string representing the sudoku or None if it fails
        """
        h, w = self.image.shape

        sizes = [600., 1200., 1800.]

        i = 0
        ratio = -1
        grid = None
        while i < 3 and ratio < 1 and grid is None:
            self.step = i
            ratio = sizes[i] / min(h, w)
            ratio = ratio if ratio < 1 else 1
            logging.info('%d try to resize', i)
            resized = cv2.resize(self.image, None, fx=ratio, fy=ratio,
                                 interpolation=cv2.INTER_CUBIC)
            grid = self.extract_grid(resized, label_tries=label_tries)
            i += 1

        if grid is not None:
            preds = self.extract_digits(grid)
            string = [' ']*81
            probs = 0
            digits = 0
            for i in range(9):
                for j in range(9):
                    val = preds[i, j]
                    if val is not 0:
                        string[j*9 + i] = str(int(val[0, 1]))
                        probs += val[0, 0]
                        digits += 1
            if digits > 0 and probs / digits > 0.9:
                return ''.join(string)
        else:
            logging.info('No grid found')

        logging.info(self.filename)
        return None

    def extract_grid(self, image, label_tries=4):
        """
        Extract the sudoku's grid from the given image
        :type image: numpy.array
        :param label_tries: number of times it tries to find a grid in the image
        :return: thresholded and polished area of the image where the sudoku is
            supposed to be (only numbers should remain)
        """
        # Edge Detection
        edge_x = cv2.Sobel(image, cv2.CV_8U, 1, 0, WSIZE[self.step])
        edge_y = cv2.Sobel(image, cv2.CV_8U, 0, 1, WSIZE[self.step])
        sobel = edge_x / 2 + edge_y / 2

        # Otsu Thresholding
        _, bw = cv2.threshold(sobel, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Closing to fill gaps (connect grid)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ONES(WSIZE[self.step]))

        # 8way labelling
        labeled, features = label(bw, structure=ONES(3))
        lbls = np.arange(1, features + 1)
        # Feature extraction: area of the label
        # The grid is supposed to be one of the greater one given its edges
        areas = extract_feature(bw, labeled, lbls, np.sum, np.uint32, 0)

        vertices = None
        i = 0

        # Repeat until we're out of tries or we found the vertices
        while i in range(0, label_tries) and vertices is None:
            logging.info('%d try to find grid', i+1)
            index = np.argmax(areas) + 1
            # Isolate label we're currently testing
            test = labeled.copy().astype(np.uint8)
            test[labeled != index] = 0
            test[labeled == index] = 255
            if self.is_grid(test, image):
                vertices = self.extract_corners(test)
                if vertices is None:
                    areas[index - 1] = 0
            else:
                # If it failed, remove this label
                logging.info('Failed to find grid')
                areas[index - 1] = 0
            i += 1

        if vertices is None:
            return None

        square = np.float32([[0, 0], [0, SSIZE],
                             [SSIZE, SSIZE], [SSIZE, 0]])

        ratio = np.size(self.image, 0) / float(np.size(image, 0))
        vertices = vertices * ratio
        m = cv2.getPerspectiveTransform(vertices.astype(np.float32), square)
        warped = cv2.warpPerspective(self.image, m, (SSIZE, SSIZE))
        ret = 255 - cv2.adaptiveThreshold(warped, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          SSIZE / 9, 10)

        self.remove_artifacts(ret)
        return ret

    def is_grid(self, grid, image):
        """
        Checks the "gridness" by analyzing the results of a hough transform.
        :param grid: binary image
        :return: wheter the object in the image might be a grid or not
        """
        #   - Distance resolution = 1 pixel
        #   - Angle resolution = 1° degree for high line density
        #   - Threshold = 144 hough intersections
        #        8px digit + 3*2px white + 2*1px border = 16px per cell
        #           => 144x144 grid
        #        144 - minimum number of points on the same line
        #       (but due to imperfections in the binarized image it's highly
        #        improbable to detect a 144x144 grid)

        lines = cv2.HoughLines(grid, 1, np.pi / 180, 144)

        if lines is not None and np.size(lines) >= 20:
            lines = lines.reshape((lines.size/2), 2)
            # theta in [0, pi] (theta > pi => rho < 0)
            # normalise theta in [-pi, pi] and negatives rho
            lines[lines[:, 0] < 0, 1] -= np.pi
            lines[lines[:, 0] < 0, 0] *= -1

            criteria = (cv2.TERM_CRITERIA_EPS, 0, 0.01)
            # split lines into 2 groups to check whether they're perpendicular
            if cv2.__version__[0] == '2':
                density, clmap, centers = cv2.kmeans(
                        lines[:, 1], 2, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            else:
                density, clmap, centers = cv2.kmeans(
                    lines[:, 1], 2, None, criteria,
                    5, cv2.KMEANS_RANDOM_CENTERS)

            # Overall variance from respective centers
            var = density / np.size(clmap)
            sin = abs(np.sin(centers[0] - centers[1]))
            # It is probably a grid only if:
            #   - centroids difference is almost a 90° angle (+-15° limit)
            #   - variance is less than 5° (keeping in mind surface distortions)
            return sin > 0.99 and var <= (5*np.pi / 180) ** 2
        else:
            return False

    def extract_corners(self, image):
        """
        Find the 4 corners of a binary image
        :param image: binary image
        :return: 4 main vertices or None
        """
        cnts, _ = cv2.findContours(image.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cnt = cnts[0]
        _, _, h, w = cv2.boundingRect(cnt)
        epsilon = min(h, w) * 0.5
        vertices = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = cv2.convexHull(vertices, clockwise=True)
        vertices = self.correct_vertices(vertices)

        return vertices

    def correct_vertices(self, pts):
        """
        Extract the main 4 vertices which could represent a square
        (within perspective limits)
        :param pts: vertices
        :return: 4 main vertices or None
        """
        if pts is None or np.size(pts, 0) < 4:
            # Even with 3 vertices it's quite difficult to retrieve the 4th one
            logging.info('Not enough vertices')
            return None

        k = np.size(pts, 0)
        angles = [compute_angle(pts[i % k], pts[(i - 1) % k], pts[(i + 1) % k])
                  for i in range(k)]

        if k >= 5:
            # Get four main angles
            outer = sorted(angles)[:4]
            # And retrieve them from original order
            angles = [angle if angle in outer else 0 for angle in angles]
            # to find their respective vertices
            vertices = np.array([pts[i] for i in range(k) if angles[i] != 0])
        else:  # k==4
            vertices = np.array(pts)

        if self.perspective:
            # Perspective math.
            pass
        else:
            # Mean of ratios between two values
            def mratio(a, b):
                return (a/b + b/a) / 2.

            lines = [compute_line(vertices[i % 4], vertices[(i + 1) % 4])
                     for i in range(4)]
            # If the mratio of any consecutive side is greater than 1.03 than,
            # even if with some perspective distortion, we extracted a wrong
            # grid or just an incomplete one.
            # 1.083 = mratio(9,8)
            if np.any(np.array([mratio(lines[i % 4], lines[(i + 1) % 4])
                                for i in range(0, 4)]) > 1.083):
                logging.info('MRATIO higher than allowed')
                return None

        return vertices

    def remove_artifacts(self, image):
        """
        Remove the connected components that are not within the parameters
        Operates in place
        :param image: sudoku's thresholded image w/o grid
        :return: None
        """
        labeled, features = label(image, structure=CROSS)
        lbls = np.arange(1, features + 1)
        areas = extract_feature(image, labeled, lbls, np.sum,
                                np.uint32, 0)
        sides = extract_feature(image, labeled, lbls, min_side,
                                np.float32, 0, True)
        diags = extract_feature(image, labeled, lbls, diagonal,
                                np.float32, 0, True)

        for index in lbls:
            area = areas[index - 1] / 255
            side = sides[index - 1]
            diag = diags[index - 1]
            if side < 5 or side > 20 \
                    or diag < 15 or diag > 25 \
                    or area < 40:
                image[labeled == index] = 0
        return None

    def extract_digits(self, image):
        """
        Extract digits from a binary image representing a sudoku
        :param image: binary image/sudoku
        :return: array of digits and their probabilities
        """
        prob = np.zeros(4, dtype=np.float32)
        digits = np.zeros((4, 9, 9), dtype=object)
        for i in range(4):
            labeled, features = label(image, structure=CROSS)
            objs = find_objects(labeled)
            for obj in objs:
                roi = image[obj]
                # center of bounding box
                cy = (obj[0].stop + obj[0].start) / 2
                cx = (obj[1].stop + obj[1].start) / 2
                dists = cdist([[cy, cx]], CENTROIDS, 'euclidean')
                pos = np.argmin(dists)
                cy, cx = pos % 9, pos / 9
                # 28x28 image, center relative to sudoku
                prediction = self.classifier.classify(morph(roi))
                if digits[i, cy, cx] is 0:
                    # Newly found digit
                    digits[i, cy, cx] = prediction
                    prob[i] += prediction[0, 0]
                elif prediction[0, 0] > digits[i, cy, cx][0, 0]:
                    # Overlapping! (noise), choose the most probable prediction
                    prob[i] -= digits[i, cy, cx][0, 0]
                    digits[i, cy, cx] = prediction
                    prob[i] += prediction[0, 0]
            image = np.rot90(image)
        logging.info(prob)
        return digits[np.argmax(prob)]
