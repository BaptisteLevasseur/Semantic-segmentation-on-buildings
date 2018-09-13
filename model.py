# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:29:24 2018

@author: HP
"""
import json
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D,concatenate

from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import resnet50


#model = ResNet50(weights='imagenet',input_shape=(224, 224, 3))

class Model_NN(object):
    def __init__(self,name):
        self.name = name
        self.model = None

    
    def model(self):
        return self.model
    
    
    def save_model(self):
        print("Saving model...")
        name_weights='models/'+self.name+'.h5'
        name_model='models/'+self.name+'.json'        
        self.model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(self.model.to_json(), outfile)
    
    def load_model(self):
        name_weights='models/'+self.name+'.h5'
        name_model='models/'+self.name+'.json'
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile("adam", "binary_crossentropy")
        self.model = model
        return self
    
class UNET_VGG(Model_NN):
    def __init__(self):
        super(UNET_VGG,self).__init__("unet_vgg16")
        self.model = self.unet_vgg16()
    
    
    def unet_vgg16(self):
        """
        U-Net architecture based on the pre-trained network VGG16.
        The encoding part is the same that the VGG16. The weights of the 18 first
        layers (encoder part) must be initialized after compilation of the model.
        (see test_setting_weight function)
        
        Return
        ------
        model : keras model
            Keras model for the U-Net structure
        """
        
        # =============================================================================
        # Encoder part
        # =============================================================================
        i   = Input((224,224,3))
        c1  = Conv2D(64,(3,3),activation='relu',padding='same')(i)
        c2  = Conv2D(64,(3,3),activation='relu',padding='same')(c1)
        mp1 = MaxPooling2D((2,2))(c2)
        c3  = Conv2D(128,(3,3),activation='relu',padding='same')(mp1)
        c4  = Conv2D(128,(3,3),activation='relu',padding='same')(c3)
        mp2 = MaxPooling2D((2,2))(c4)
        c5  = Conv2D(256,(3,3),activation='relu',padding='same')(mp2)
        c6  = Conv2D(256,(3,3),activation='relu',padding='same')(c5)
        c7  = Conv2D(256,(3,3),activation='relu',padding='same')(c6)
        mp3 = MaxPooling2D((2,2))(c7)
        c8  = Conv2D(512,(3,3),activation='relu',padding='same')(mp3)
        c9  = Conv2D(512,(3,3),activation='relu',padding='same')(c8)
        c10 = Conv2D(512,(3,3),activation='relu',padding='same')(c9)
        mp4 = MaxPooling2D((2,2))(c10)
        c11  = Conv2D(512,(3,3),activation='relu',padding='same')(mp4)
        c12  = Conv2D(512,(3,3),activation='relu',padding='same')(c11)
        c13 = Conv2D(512,(3,3),activation='relu',padding='same')(c12)
        mp5 = MaxPooling2D((2,2))(c13)
        # =============================================================================
        # Center part
        # =============================================================================
        c14 = Conv2D(512,(3,3),activation='relu',padding='same')(mp5)
        # =============================================================================
        # Decoder part
        # =============================================================================
        d1 = Conv2DTranspose(512,(3,3),activation='relu',padding='same',strides=2)(c14)  # or 256?
        m1 = concatenate([d1,c13])
        c15 = Conv2D(512,(3,3),activation='relu',padding='same')(m1)
        d2 = Conv2DTranspose(256,(3,3),activation='relu',padding='same',strides=2)(c15)
        m2 = concatenate([d2,c10])
        c16 = Conv2D(512,(3,3),activation='relu',padding='same')(m2)
        d3 = Conv2DTranspose(128,(3,3),activation='relu',padding='same',strides=2)(c16)
        m3 = concatenate([d3,c7])
        c17 = Conv2D(256,(3,3),activation='relu',padding='same')(m3)
        d4 = Conv2DTranspose(64,(3,3),activation='relu',padding='same',strides=2)(c17)
        m4 = concatenate([d4,c4])
        c18 = Conv2D(128,(3,3),activation='relu',padding='same')(m4)
        d5 = Conv2DTranspose(32,(3,3),activation='relu',padding='same',strides=2)(c18)
        m5 = concatenate([d5,c2])
        c19 = Conv2D(1,(3,3),activation='sigmoid',padding='same')(m5)
        
        model = Model(inputs=i,outputs=c19)
        return model


class FCN(Model_NN):
    def __init__(self):
        super(FCN,self).__init__("fcn")
        self.model = self.fcn()
    
    def fcn(self):
        # =============================================================================
        # Encoder part
        # =============================================================================
        i   = Input((224,224,3))
        c1 = Conv2D(16,(3,3),activation='relu',padding='same')(i)
        c2 = Conv2D(32,(3,3),activation='relu',padding='same')(c1)
        mp1 = MaxPooling2D((2,2))(c2)
        c3 = Conv2D(16,(3,3),activation='relu',padding='same')(mp1)
        d1 = Conv2DTranspose(16,(3,3),activation='relu',padding='same',strides=2)(c3)
        c4 = Conv2D(1,(3,3),activation='sigmoid',padding='same')(d1)
        
        model = Model(inputs=i,outputs=c4)
        return model
    


def test_setting_weight():
    model = UNET_VGG()
    
    model_vgg = VGG16(weights='imagenet',input_shape=(224, 224, 3))
    model_vgg.summary()
    
    # We must init weight after the compilation
    model.model.compile(loss= 'categorical_crossentropy',
                        optimizer='adam',metrics=['accuracy'])
    
    # Initializing weight with trained vgg16, 18 = size of the encoder part
    for i in range(18): 
        model.model.layers[i].set_weights(model_vgg.layers[i].get_weights())
    
    # Check model structure
    model.model.summary()
    
    
if __name__=="__main__":
    test_setting_weight()
    model = FCN()
    model.summary()