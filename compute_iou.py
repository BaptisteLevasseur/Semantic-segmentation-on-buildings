# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:11:36 2018

@author: HP
"""

import buzzard as buzz
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, join
from os import listdir



def poly_to_binary(file,
                   gt_train = "AerialImageDataset/train/gt/",
                   polygons_path = "geoJSON/",downsampling_factor = 4):
    ds = buzz.DataSource(allow_interpolation=True)
    ds.open_raster('binary', gt_train+file) 
    geojson_file = polygons_path+file.split('.')[0]+'.geojson'
    ds.open_vector('polygons',geojson_file,driver='geoJSON')
    
    fp = buzz.Footprint(
            tl=ds.binary.fp.tl,
            size=ds.binary.fp.size,
            rsize=ds.binary.fp.rsize/downsampling_factor ,
        )
    
    binary = ds.binary.get_data(band=(1), fp=fp).astype('uint8')
    binary_predict = np.zeros_like(binary)
    
    for poly in ds.polygons.iter_data(None):
        mark_poly = fp.burn_polygons(poly)
        binary_predict[mark_poly] = 1
    return binary,binary_predict

def iou(binary,binary_pred):
    b = binary.astype('bool')
    bp = binary_pred.astype('bool')
    inter = np.logical_and(b,bp)
    union = np.logical_or(b,bp)
    return np.sum(inter)/np.sum(union)

def compute_iou(images_train,downsampling_factor=4):
    files = [f for f in listdir(images_train) if isfile(join(images_train, f))]
    list_iou = np.zeros((len(files)))
    for i in range(len(files)):
        print("Processing file number "+str(i)+"/"+str(len(files))+"...")
        b,bp=poly_to_binary(files[i], downsampling_factor = downsampling_factor)
        list_iou[i] = iou(b,bp)
        print(list_iou[i])
    np.save('Results/iou',list_iou)
    return list_iou
        
def display_stats():
    results = np.load('Results/iou.npy')*100.
    names = ['austin','chicago','kitsap','tyrol-w','vienna']
    size = 36
    mean = np.zeros(len(names))
    var = np.zeros(len(names))
    mini = np.zeros(len(names))
    maxi = np.zeros(len(names))
    for i in range(len(names)):
        results_i = results[size*i:size*(i+1)]
        mean[i] = np.mean(results_i)
        var[i] = np.var(results_i)
        maxi[i] = np.max(results_i)
        mini[i] = np.min(results_i)
    print(names)
    print(mean)
    print(var)
    print(maxi)
    print(mini)
    print(np.argmax(results))
images_train = "AerialImageDataset/train/images/"
gt_train = "AerialImageDataset/train/gt/"
polygons_path = "geoJSON/"
#compute_iou(images_train,downsampling_factor=4)


display_stats()


