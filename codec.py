import time

import numpy as np
from keras import Sequential, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Conv2DTranspose, Flatten, LeakyReLU, Reshape
from keras.optimizers import Nadam

import image_compression_lib as icl

from keras import backend as K
import tensorflow as tf

class imgcodec:

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    def __init__(self):
        self.x_train = self.normalize(self.x_train)
        self.x_test = self.normalize(self.x_test)

        self.y_train = self.x_train
        self.y_test = self.x_test

        self.input_shape = self.x_train.shape[1::]
        self.features = 6
        self.optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.loss = [self.__binary_focal_loss()]

        self.coder = self.__coder()
        self.coder.compile(loss=self.loss, optimizer=self.optimizer)

        self.decoder = self.__decoder()
        self.decoder.compile(loss=self.loss, optimizer=self.optimizer)

        self.codec = self.__codec()

    def normalize(self, data):
        out = data / 255.0

        out = np.reshape(out, [-1, out.shape[1], out.shape[2], 1])
        return out

    def __binary_focal_loss(self, alpha=.9, gamma=5.):
        """
        Binary form of focal loss.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
        model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        def binary_focal_loss_fixed(y_true, y_pred):
            """
            :param y_true: A tensor of the same shape as `y_pred`
            :param y_pred:  A tensor resulting from a sigmoid
            :return: Output tensor.
            """
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

            epsilon = K.epsilon()
            # clip to prevent NaN's and Inf's
            pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
            pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return binary_focal_loss_fixed

    def __coder(self):
        model = Sequential()
        model.add(Conv2D(filters=self.features, kernel_size=9, strides=2, padding="same",
                         input_shape=self.input_shape))
        model.add(LeakyReLU())

        model.add(Conv2D(filters=self.features,
                         kernel_size=5, strides=2, padding="same"))

        model.add(Flatten())

        return model

    def __decoder(self):
        model = Sequential()

        model.add(Reshape((7, 7, self.features)))

        model.add(Conv2DTranspose(filters=self.features,
                                  kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(
            filters=1, kernel_size=9, strides=2, padding="same"))
        model.add(LeakyReLU())

        return model

    def __codec(self):
        model = Sequential()

        model.add(self.coder)
        model.add(self.decoder)

        model.summary()

        return model

    def __custom_loss(self):

        pass

    def codec_training(self):
        reduceLR = ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=2,
            verbose=1,
            cooldown=2)
        earlyStop = EarlyStopping(
            monitor='loss',
            patience=10,
            verbose=1,
            restore_best_weights=True)

        callbacks = [reduceLR, earlyStop]

        self.codec.compile(loss=self.loss, optimizer=self.optimizer)
        self.codec.fit(
            self.x_train, self.y_train,
            batch_size=2048,
            epochs=10000000,
            callbacks=callbacks,
            use_multiprocessing=True,
            verbose=1,
            validation_data=[self.x_test, self.y_test])

        save_path = "./models/"
        icl.folder_maker(save_path)
        self.coder.save(save_path+"coder-" +
                        time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")
        self.decoder.save(
            save_path+"decoder-"+time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")
        self.codec.save(save_path+"codec-" +
                        time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")


if __name__ == "__main__":
    codec = imgcodec()
    codec.codec_training()

    # i=0
    N = 20
    for i in range(N):
        test = np.reshape(codec.x_test[i], [28, 28])
        gerner = codec.coder.predict(
            np.reshape(codec.x_test[i], [1, 28, 28, 1]))
        gerner = codec.decoder.predict(np.reshape(gerner, [1, 196]))
        gerner = np.reshape(gerner, [28, 28])
        icl.picture(test, gerner, str(i).zfill(3))
        print(str(i)+"/"+str(N)+"\r", end="")
