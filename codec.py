import time

import numpy as np
from keras import Sequential, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Conv2DTranspose, Flatten, LeakyReLU, Reshape, BatchNormalization, Dense, AveragePooling2D, UpSampling2D
from keras.optimizers import Nadam, Adam
from keras.models import load_model
from keras.losses import mean_squared_error, binary_crossentropy

import image_compression_lib as icl

from keras import backend as K
import tensorflow as tf


class imgcodec:

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    def __init__(self):
        self.x_train = icl.normalize(self.x_train)
        self.x_test = icl.normalize(self.x_test)

        self.y_train = self.x_train
        self.y_test = self.x_test

        self.input_shape = self.x_train.shape[1::]
        self.features = 8
        self.coding_stream = 150
        self.optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        # self.loss = [icl.binary_focal_loss()]
        self.loss = mean_squared_error

        self.coder = self.__coder()
        self.coder.compile(loss=self.loss, optimizer=self.optimizer)

        self.decoder = self.__decoder()
        self.decoder.compile(loss=self.loss, optimizer=self.optimizer)

        self.codec = self.__codec()

    def __coder(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=5, strides=1, padding="same",
                         input_shape=self.input_shape))
        model.add(LeakyReLU())
        model.add(AveragePooling2D(pool_size=2, padding="same"))

        model.add(Conv2D(filters=4, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(AveragePooling2D(pool_size=2, padding="same"))

        model.add(Flatten())

        self.coding_stream = model.output_shape

        return model

    def __decoder(self):
        model = Sequential()
        model.add(Reshape((7, 7, 4), input_shape=self.coding_stream[1::]))

        model.add(Conv2D(filters=4, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(UpSampling2D(size=(2, 2), interpolation="bilinear"))

        model.add(Conv2D(filters=8, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(UpSampling2D(size=(2, 2), interpolation="bilinear"))

        model.add(Conv2D(filters=1, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())

        return model

    def __codec(self):
        model = Sequential()

        model.add(self.coder)
        model.add(self.decoder)

        model.summary()

        return model

    def codec_trainer(self):
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
            batch_size=8192,
            epochs=10000000,
            callbacks=callbacks,
            use_multiprocessing=True,
            verbose=1,
            validation_data=[self.x_test, self.y_test])

        save_path = "./models/"
        icl.folder_maker(save_path)
        self.coder.save(save_path+"coder-" + time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")
        self.decoder.save(save_path+"decoder-"+time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")
        self.codec.save(save_path+"codec-" + time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")


if __name__ == "__main__":
    codec = imgcodec()
    codec.codec_trainer()
    # codec.coder = load_model("./models/coder-2020_02_11-09_30.h5", custom_objects={"binary_focal_loss_fixed":icl.binary_focal_loss_fixed, "binary_focal_loss":icl.binary_focal_loss})
    # codec.decoder = load_model("./models/decoder-2020_02_11-09_30.h5", custom_objects={"binary_focal_loss_fixed":icl.binary_focal_loss_fixed, "binary_focal_loss":icl.binary_focal_loss})

    N = 20
    for i in range(N):
        test = np.reshape(codec.x_test[i], [28, 28])
        gerner = codec.coder.predict(np.reshape(codec.x_test[i], [1, 28, 28, 1]))
        gerner = codec.decoder.predict(np.reshape(gerner, [1, -1]))
        gerner = icl.unormalize(gerner)
        icl.picture(test, gerner, str(i).zfill(3))
        print(str(i)+"/"+str(N)+"\r", end="")
