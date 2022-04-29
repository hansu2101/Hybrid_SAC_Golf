import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def deg_to_rad(deg):
    return deg / 180.0 * math.pi


def rotation_2d(rot):
    return np.array([[math.cos(rot), -math.sin(rot)],
                     [math.sin(rot), math.cos(rot)]])


def transform_2d(tr_x, tr_y, rot):
    return np.array([[math.cos(rot), -math.sin(rot), tr_x],
                     [math.sin(rot), math.cos(rot), tr_y],
                     [0, 0, 1]])


def inv_transform_2d(tf):
    assert (tf.shape == (3, 3))

    rot = tf[:2, :2]
    translation = tf[:2, 2:3]
    inv = np.concatenate([rot.transpose(), np.dot(-rot.transpose(), translation)], 1)
    inv = np.concatenate([inv, np.array([[0, 0, 1]])])
    return inv


def is_within(bounds_min, bounds_max, x, include_equality=True):
    assert (len(bounds_min) == len(bounds_max) == len(x))

    for i in range(len(x)):
        if include_equality:
            if x[i] <= bounds_min[i] or bounds_max[i] <= x[i]:
                return False
        else:
            if x[i] < bounds_min[i] or bounds_max[i] < x[i]:
                return False

    return True


def show_grayscale(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    plt.show()
