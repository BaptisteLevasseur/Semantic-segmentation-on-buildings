# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 01:20:54 2018

@author: HP
"""

from model import UNET_VGG,FCN,FCN2
import generator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from metrics import MeanIoU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


import matplotlib.pyplot as plt
import numpy as np
import pickle


def train(model_nn, dataset,
          epochs=10, batch_size=1000,
          learning_rate=1e-4, beta_1 = 0.9, beta_2= 0.999, initialize_weights_vgg = False, load_weights=False):
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
    if load_weights:
        model_nn.model.load_weights("weights.best.hdf5")
    
    if initialize_weights_vgg:
        print("Loading weights...")
        # Initializing weight with trained vgg16, 18 = size of the encoder part
        model_vgg = VGG16(weights='imagenet',input_shape=(224, 224, 3))
        for i in range(18): 
            model_nn.model.layers[i].set_weights(model_vgg.layers[i].get_weights())
        
    print("Training model...")
    
    # checkpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    history = model_nn.model.fit(X_train, Y_train,
                                 epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test,Y_test),
                                 callbacks=callbacks_list) 
    
    # Save weights
    print("Saving model...")
    model_nn.save_model()
    return model_nn, history


if __name__ == '__main__':
    image_size = 128
    load_dataset=True
    
    if load_dataset:
        with open('dataset.pickle', 'rb') as f:
            dataset = pickle.load(f)
            (X_train, Y_train, X_test, Y_test) = dataset 
    else:
        print("Generating train set")
        X_train,Y_train = generator.compute_dataset(image_size=image_size,
                                                    n_files = None,
                                                    downsampling_factor=3, 
                                                    n_samples=5)
        print("Generating test set")
        X_test,Y_test = generator.compute_dataset(image_size=image_size,
                                                  n_files = None, 
                                                  downsampling_factor=3)
        dataset = (X_train, Y_train, X_test, Y_test)
        with open('dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)
    
    print("Loading model")
#    model = UNET_VGG()
    model_nn = FCN2(image_size)
    model_nn,history = train(model_nn, dataset, epochs=50, batch_size=40,
                             learning_rate = 1e-4, initialize_weights_vgg=False)
    
#    model_nn = load_model('save_temp/fcn2_128.h5') 
    test = model_nn.predict(X_test[0].reshape(1,image_size,image_size,3))

    
    
    # summarize history for accuracy
    np.save('history/loss_'+model_nn.name+'_'+str(image_size),
            history.history['loss'])
    np.save('history/val_loss_'+model_nn.name+'_'+str(image_size),
            history.history['val_loss'])
    
    plt.imshow(test.reshape((image_size,image_size)))
    plt.show()

    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
