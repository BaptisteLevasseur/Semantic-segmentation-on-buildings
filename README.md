# Semantic segmentation on buildings

### Project 

This project is based on the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).
The aim is to build an algorithm based on neural network performing semantic segmentation on buildings.

The open source library [buzzard](https://github.com/airware/buzzard/) developped by Airware has been used to manage georeferenced data. 
The results are saved and store at the geoJSON format.

### Main ideas

The solution implemented is inspired by the [TernausNet](https://arxiv.org/abs/1801.05746). It is a autoencoder that reuses the features learned during the encoding part for the decoding part.

__________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            (None, 128, 128, 3)  0                                            
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 128, 128, 64) 1792        input_5[0][0]                    
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 128, 128, 64) 36928       conv2d_25[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_9 (MaxPooling2D)  (None, 64, 64, 64)   0           conv2d_26[0][0]                  
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 64, 64, 128)  73856       max_pooling2d_9[0][0]            
__________________________________________________________________________________________________
max_pooling2d_10 (MaxPooling2D) (None, 32, 32, 128)  0           conv2d_27[0][0]                  
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 32, 32, 128)  147584      max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_transpose_9 (Conv2DTrans (None, 64, 64, 64)   73792       conv2d_28[0][0]                  
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 64, 64, 192)  0           conv2d_transpose_9[0][0]         
                                                                 conv2d_27[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_10 (Conv2DTran (None, 128, 128, 32) 55328       concatenate_9[0][0]              
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 128, 128, 96) 0           conv2d_transpose_10[0][0]        
                                                                 conv2d_26[0][0]                  
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 128, 128, 64) 55360       concatenate_10[0][0]             
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 128, 128, 1)  577         conv2d_29[0][0]                  
==================================================================================================
Total params: 445,217
Trainable params: 445,217
Non-trainable params: 0

### Resuts

