# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 01:20:54 2018

@author: HP
"""

from model import UNET_VGG,FCN
import generator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from metrics import MeanIoU

import matplotlib.pyplot as plt
import numpy as np


def train(model_nn, dataset,
          epochs=10, batch_size=1000,
          learning_rate=1e-4, beta_1 = 0.9, beta_2= 0.999, initialize_weights = False):
    """
    Train the model to fit the dataset
    Parameters
    ----------
    model_nn : model to be trained (Model_NN inherited class)
    dataset : tuple (X_train,Y_train,X_test,Y_test)
    initialize_weights : used for the vgg16 pretrained network if true
    """
    X_train,Y_train,X_test,Y_test = dataset
    
    print("Setting model parameters...")
    
    adam_optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2)
    miou_metric = MeanIoU(2)
    model_nn.model.compile(loss= 'binary_crossentropy',
                           optimizer=adam_optimizer)
    
    if initialize_weights:
        print("Loading weights...")
        # Initializing weight with trained vgg16, 18 = size of the encoder part
        model_vgg = VGG16(weights='imagenet',input_shape=(224, 224, 3))
        for i in range(18): 
            model_nn.model.layers[i].set_weights(model_vgg.layers[i].get_weights())
        
    print("Training model...")
    history = model_nn.model.fit(X_train, Y_train,
                                 epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test,Y_test)) 
    
    # Save weights
    print("Saving model...")
    model_nn.save_model()
    return model_nn, history


if __name__ == '__main__':
    print("Generating train set")
    X_train,Y_train = generator.compute_dataset(image_size=224,
                                                n_files = None,
                                                downsampling_factor=2, 
                                                n_samples=30)
    print("Generating test set")
    X_test,Y_test = generator.compute_dataset(image_size=224,
                                              n_files = None, 
                                              downsampling_factor=2)
    dataset = (X_train, Y_train, X_test, Y_test)
    
    print("Loading model")
#    model = UNET_VGG()
    model_nn = FCN()
    model_nn,history = train(model_nn, dataset, epochs=20, batch_size=100,
                             learning_rate = 1e-4, initialize_weights=False)
    model_nn.model.predict(X_test[1].reshape(1,224,224,3))
    
        # summarize history for accuracy
    np.save('history_loss',history.history['loss'])
    np.save('history_val_loss',history.history['val_loss'])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
