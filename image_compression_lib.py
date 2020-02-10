import os
import time

import numpy as np
from matplotlib import pyplot as plt

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