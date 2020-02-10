import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Reshape
from keras import losses
from keras.optimizers import Nadam
from matplotlib import pyplot as plt
import os
import time
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
        self.features = 4
        self.optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.loss = losses.mean_squared_error

        self.coder = self.__coder()
        self.coder.compile(loss=self.loss, optimizer=self.optimizer)

        self.decoder = self.__decoder()
        self.decoder.compile(loss=self.loss, optimizer=self.optimizer)

        self.codec = self.__codec()

    def normalize(self, data):
        out = data / 255.0

        out = np.reshape(out, [-1, out.shape[1], out.shape[2], 1])
        return out

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
            cooldown=0)
        earlyStop = EarlyStopping(
            monitor='loss',
            patience=5,
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
        folder_maker(save_path)
        self.coder.save(save_path+"coder-" +
                        time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")
        self.decoder.save(
            save_path+"decoder-"+time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")
        self.codec.save(save_path+"codec-" +
                        time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")

def focal_loss(alpha=0.9, gamma=2.0):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

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

if __name__ == "__main__":
    imgcodec = imgcodec()
    imgcodec.codec_training()

    # i=0
    N = 20
    for i in range(N):
        test = np.reshape(imgcodec.x_test[i], [28, 28])
        gerner = imgcodec.coder.predict(
            np.reshape(imgcodec.x_test[i], [1, 28, 28, 1]))
        gerner = imgcodec.decoder.predict(np.reshape(gerner, [1, 196]))
        gerner = np.reshape(gerner, [28, 28])
        picture(test, gerner, str(i).zfill(3))
        print(str(i)+"/"+str(N)+"\r", end="")
