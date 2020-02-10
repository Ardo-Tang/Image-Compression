import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
from keras.losses import mean_squared_error
from keras.optimizers import Nadam
from matplotlib import pyplot as plt
import os
import time 


class imgcodec:
    
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    def __init__(self):
        self.x_train = self.normalize(self.x_train)
        self.x_test = self.normalize(self.x_test)

        self.y_train = self.x_train
        self.y_test = self.x_test

        self.input_shape = self.x_train.shape[1::]
        self.codec = self.__codec_model()

    def normalize(self, data):
        out = data / 255.0

        out = np.reshape(out, [-1, out.shape[1], out.shape[2], 1])
        return out

    def __codec_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=9, strides=2, padding="same", input_shape=self.input_shape))#, input_shape=self.input_shape
        model.add(LeakyReLU())

        model.add(Conv2D(filters=32, kernel_size=5, strides=2, padding="same"))
        
        #coder output
        #quantize
        #decoder input

        model.add(Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(filters=1, kernel_size=9, strides=2, padding="same"))
        model.add(LeakyReLU())
        
        model.summary()

        return model

    def codec_training(self):
        reduceLR = ReduceLROnPlateau(
            monitor='loss', 
            factor=0.5, 
            patience=3, 
            verbose=1,
            cooldown=0)
        earlyStop = EarlyStopping(
            monitor='loss', 
            patience=10, 
            verbose=1,
            restore_best_weights=True)

        callbacks = [reduceLR, earlyStop]

        optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.codec.compile(loss=mean_squared_error, optimizer=optimizer)
        self.codec.fit(
            self.x_train, self.y_train, 
            batch_size=64,
            epochs=2,
            callbacks=callbacks,
            use_multiprocessing=True,
            verbose=1,
            validation_data=[self.x_test,self.y_test])

def picture(x_test, x_gener):
    save_path = "./output_img/"
    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)
    plt.figure()
    plt.subplot(211)
    plt.imshow(x_test)
    plt.subplot(212)
    plt.imshow(x_gener)
    plt.savefig(save_path+time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".png")

if __name__ == "__main__":
    imgcodec = imgcodec()
    imgcodec.codec_training()

    picture(imgcodec.x_test[0], imgcodec.codec.predict(imgcodec.x_test[0]))