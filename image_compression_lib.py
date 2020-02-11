import os
import time

import numpy as np
from matplotlib import pyplot as plt

from keras import backend as K
import tensorflow as tf

def binary_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    alpha=.9
    gamma=5.

    pt_1 = tf.where(tf.equal(y_true, tf.ones_like(y_true)), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, tf.zeros_like(y_true)), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
        -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def binary_focal_loss():
    """
    Binary form of focal loss.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    return binary_focal_loss_fixed

def folder_maker(folder_name):
    if(not os.path.isdir(folder_name)):
        os.mkdir(folder_name)

def picture(x_test, x_gener, index):
    save_path = "./output_img/"
    folder_maker(save_path)

    plt.figure()
    plt.subplot(211)
    plt.imshow(x_test)
    plt.subplot(212)
    plt.imshow(x_gener)
    plt.savefig(save_path+time.strftime("%Y_%m_%d-%H_%M",
                                        time.localtime())+"-"+index+".png")
    plt.close()