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
            patience=2, 
            verbose=1,
            cooldown=0)
        earlyStop = EarlyStopping(
            monitor='loss', 
            patience=5, 
            verbose=1,
            restore_best_weights=True)

        callbacks = [reduceLR, earlyStop]

        optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.codec.compile(loss=mean_squared_error, optimizer=optimizer)
        self.codec.fit(
            self.x_train, self.y_train, 
            batch_size=2048,
            epochs=10000000,
            callbacks=callbacks,
            use_multiprocessing=True,
            verbose=1,
            validation_data=[self.x_test,self.y_test])

        save_path = "./model/"
        folder_maker(save_path)
        self.codec.save(save_path+time.strftime("%Y_%m_%d-%H_%M", time.localtime())+".h5")

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
    plt.savefig(save_path+time.strftime("%Y_%m_%d-%H_%M", time.localtime())+"-"+index+".png")

if __name__ == "__main__":
    imgcodec = imgcodec()
    imgcodec.codec_training()

    # i=0
    for i in range(500):
        test = np.reshape(imgcodec.x_test[i], [28,28])
        gerner = np.reshape(imgcodec.codec.predict(np.reshape(imgcodec.x_test[i], [1,28,28,1])), [28,28])
        picture(test, gerner, i.zfill(3))