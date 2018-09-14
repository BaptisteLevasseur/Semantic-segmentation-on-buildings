# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:33:30 2018

@author: HP
"""

import numpy as np
import buzzard as buzz
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import tqdm

"""
TODO : 
    Add rotation for data augmentation
    Add change of luminence
"""
def random_crop(ds,crop_rsize=700, factor=1, rotation=0):
    '''
    Returns a random crop of a geotiff buzzard datasource
    Parameters
    ----------
    ds : DataSource
        Input buzzard DataSource
    crop_size : int
        Size in pixel of the cropped final image
    factor : int
        Downsampling factor
    rotation : float
        Rotation angle (rad)
    Return
    ------
    fp : Footprint
        Cropped footprint 
    '''
    # Cropping parameters
    crop_factor = ds.rgb.fp.rsize/crop_rsize
    crop_size = ds.rgb.fp.size/crop_factor
    # Original footprint
    fp = buzz.Footprint(
        tl=ds.rgb.fp.tl,
        size=ds.rgb.fp.size,
        rsize=ds.rgb.fp.rsize ,
    )
    # New random footprint
    tl = np.array([
            np.random.randint(fp.tl[0],fp.tl[0] + fp.size[0] - factor*crop_size[0]), #x
            np.random.randint(fp.tl[1] - fp.size[1] + factor*crop_size[1], fp.tl[1]) #y
             ])
    fp = buzz.Footprint(
        tl=tl,
        size=crop_size*factor,
        rsize=[crop_rsize, crop_rsize],
    )
    return fp


def compute_dataset(image_size=448, n_files = None, downsampling_factor=1, n_samples=2):
    '''
    Computes the augmented dataset of length (n_files*n_samples).
    
    Parameters
    ----------
    image_size : int
        size of the input images for the neural network
    n_files : int
        number of images in the total inria dataset to consider (for debugging)
    downsampling_factor : float
        downsampling factor to lower the resolution
    n_samples : int
        number of cropped images to generate for each image in inria dataset
    Returns
    -------
    X_train : np.ndarray
        Input array for the neural network (n_files*samples rgb images)
    Y_train : np.ndarray
        Output array for the neural network (n_files*samples binary images)
    '''
    images_train = "AerialImageDataset/train/images/"
    gt_train = "AerialImageDataset/train/gt/"
    files = [f for f in listdir(images_train) if isfile(join(images_train, f))]
    
    if n_files is None:
        n_files = len(files)
        
    X_train = np.zeros((n_files*n_samples,image_size,image_size,3))
    Y_train = np.zeros((n_files*n_samples,image_size,image_size,1))
    for i in tqdm.tqdm(range(n_files)):
        # current image
        file = files[i]
        ds_rgb = buzz.DataSource(allow_interpolation=True)
        ds_binary = buzz.DataSource(allow_interpolation=True)
        rgb_path = images_train + file
        binary_path = gt_train + file   
        ds_rgb.open_raster('rgb', rgb_path)
        ds_binary.open_raster('rgb', binary_path)
        
        # n_samples random crop of the image
        for sample in range(n_samples):
            fp = random_crop(ds_rgb,crop_rsize=image_size,factor=downsampling_factor)
            rgb= ds_rgb.rgb.get_data(band=(1, 2, 3), fp=fp).astype('uint8')
            binary= ds_binary.rgb.get_data(band=(1), fp=fp).astype('uint8')
            # Normalization of data
            X_train[i*n_samples+sample] = rgb/127.5-1.0
            almost_binary_y=  binary.reshape((image_size,image_size,1))/127.5-1.0
            Y_train[i*n_samples+sample] = almost_binary_y >0
    return X_train,Y_train

def show_image(X_train,Y_train,n):
    fig = plt.figure()
    plt.title('Test raster')
    ax = fig.add_subplot(111)
    ax.imshow((Y_train[n,:,:,0]+1)*127.5)
    plt.show()
    fig = plt.figure()
    plt.title('Test raster')
    ax = fig.add_subplot(111)
    ax.imshow(((X_train[n,:,:,:]+1)*127.5).astype(np.uint8))
    plt.show()


if __name__== "__main__":
    X_train,Y_train=compute_dataset(image_size=224, n_files = None,
                                    downsampling_factor=2, n_samples=30)
    print(X_train.shape)







