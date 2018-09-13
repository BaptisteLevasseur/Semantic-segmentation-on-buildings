# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:40:27 2018

@author: HP
"""

import buzzard as buzz
import numpy as np

import matplotlib.pyplot as plt
import descartes

from model import FCN
import tqdm

def tile_image(tile_size, ds, fp):
    '''
    Tiles image in several tiles of size (tile_size, tile_size)
    Params
    ------
    tile_siz : int
        size of a tile in pixel (tiles are square)
    ds: Datasource
        Datasource of the input image (binary)
    fp : footprint
        global footprint 
    Returns
    -------
    rgb_array : np.ndarray
        array of dimension 5 (x number of tiles, y number of tiles,
        x size of tile, y size of tile, number of canal)
    tiles : np.ndarray of footprint
        array of of size (x number of tiles, y number of tiles) 
        that contains footprint information
    
        
    '''
    tiles = fp.tile((tile_size,tile_size))
    rgb_array = np.zeros([tiles.shape[0],tiles.shape[0],tile_size,tile_size,3],
                         dtype='uint8')
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            rgb_array[i,j] = ds.rgb.get_data(band=(1,2,3), fp=tiles[i,j])
    return rgb_array, tiles

def untile(rgb_array,tiles,fp):
    '''
    DEPRECATED
    Get the tile binary array and the footprint matrix and returns the reconstructed image
    Params
    ------
    rgb_array : np.ndarray
        array of dimension 5 (x number of tiles, y number of tiles,
        x size of tile, y size of tile)
    tiles : np.ndarray of footprint
        array of of size (x number of tiles, y number of tiles) 
        that contains footprint information
    Returns
    -------
    rgb_reconstruct : np.ndarray
        reconstructed rgb image
    '''
    # initialization of the reconstructed rgb array
    rgb_reconstruct = np.zeros([
        rgb_array.shape[0]*rgb_array.shape[2], #pixels along x axis
        rgb_array.shape[1]*rgb_array.shape[3], # pixels along y axis
        ], dtype='uint8')
    
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            tile_size = tiles[i,j].rsize
            rgb_reconstruct[i*tile_size[0]:(i+1)*tile_size[0],
                            j*tile_size[1]:(j+1)*tile_size[1]
                            ] = rgb_array[i,j]
    rgb_reconstruct = rgb_reconstruct[:fp.rsize[0],:fp.rsize[0]] # delete padding
    return rgb_reconstruct

def untile_and_predict(rgb_array,tiles,fp,model):
    '''
    Get the tile binary array and the footprint matrix and returns the reconstructed image
    Params
    ------
    rgb_array : np.ndarray
        array of dimension 5 (x number of tiles, y number of tiles,
        x size of tile, y size of tile,3)
    tiles : np.ndarray of footprint
        array of of size (x number of tiles, y number of tiles) 
        that contains footprint information
    Returns
    -------
    rgb_reconstruct : np.ndarray
        reconstructed rgb image
    '''
    # initialization of the reconstructed rgb array
    binary_reconstruct = np.zeros([
        rgb_array.shape[0]*rgb_array.shape[2], #pixels along x axis
        rgb_array.shape[1]*rgb_array.shape[3], # pixels along y axis
        ], dtype='uint8')
    
    for i in tqdm.tqdm(range(tiles.shape[0])):
        for j in range(tiles.shape[1]):
            tile_size = tiles[i,j].rsize
            # predict the binaryzed image
            predicted = predict_image(rgb_array[i,j],model_nn)
            # add the image in the global image
            binary_reconstruct[i*tile_size[0]:(i+1)*tile_size[0],
                            j*tile_size[1]:(j+1)*tile_size[1]
                            ] = predicted
    # delete the tilling padding
    binary_reconstruct = binary_reconstruct[:fp.rsize[0],:fp.rsize[0]] 
    return binary_reconstruct


def show_polygons(rgb,fp,poly_list):
    # Show image with matplotlib and descartes
    fig = plt.figure(figsize=(5. / fp.height * fp.width, 5))
    plt.title('Roof boundary')
    ax = fig.add_subplot(111)
    ax.imshow(rgb, extent=[fp.lx, fp.rx, fp.by, fp.ty])
        
    for poly in poly_list:
        ax.add_patch(descartes.PolygonPatch(poly, fill=False, ec='#ff0000', lw=3, ls='--'))
    plt.show()

def predict_image(image,model):
    '''
    Predict one image with the model. Returns a binary array
    '''
    shape_im = image.shape
    predicted_image = model.model.predict(image.reshape(1, shape_im[0], shape_im[1],3))
    predicted_image = predicted_image.reshape(shape_im[0],shape_im[1]) > 0.5
    predicted_image = predicted_image*255
    return predicted_image


def test_tile_function(ds_rgb, tiles):
    # test size of tiles
    print("Size tile in meters : " +str(tiles[0,0].size))
    print("Size tile in pixels : " +str(tiles[0,0].rsize))
    size_tiles_meters = tiles[0,0].size[0]*len(tiles)
    size_tiles_pixels= tiles[0,0].rsize[0]*len(tiles)
    print("Size total "+str(len(tiles))+ " tiles : " + str(size_tiles_meters))
    print("Size total "+str(len(tiles))+ " tiles : " + str(size_tiles_pixels))
    tile = tiles[0]
    for fp_tile in tile:
        tl = fp_tile.tl
        expected_tile = tl[0]+fp_tile.size[0]
        print(tl[0])
        print("expected :" + str(expected_tile))
    
    #une tile : 224 pixels, 67.2 m
    #23 tiles : 5152 pixels, 1545.6 -> OK
    # les tiles doivent être coupées:
    cut_rgb = ds_rgb.rgb.get_data(band=(1, 2, 3), fp=tiles[22,22]).astype('uint8')    
    plt.imshow(cut_rgb)
    plt.show()


def predict_map(model,tile_size,ds_rgb,fp):
    '''
    Pipeline from the whole rasper and footprint adapted to binary array
    Params
    ------
    model : Model object
        trained model for the prediction of image (tile_size * tile_size)
    tile_size : int
        size of tiles (i.e. size of the input array for the model)
    ds_rgb : datasource
        Datasource object for the rgb image
    fp : footprint
        footprint of the adapted image (with downsampling factor)
    '''
    print("Tiling images...")
    rgb_array, tiles = tile_image(tile_size, ds_rgb, fp)
    print("Predicting tiles..")
    predicted_binary = untile_and_predict(rgb_array,tiles,fp,model)
    print("Image binaryzed")
    return predicted_binary


if __name__=="__main__":
    # load image
    images_train = "AerialImageDataset/train/images/"
    gt_train = "AerialImageDataset/train/gt/"
    file= rgb_path = 'austin1.tif'

    ds_rgb = buzz.DataSource(allow_interpolation=True)
    ds_binary = buzz.DataSource(allow_interpolation=True)

    rgb_path = images_train + file
    binary_path = gt_train + file   
    ds_rgb.open_raster('rgb', rgb_path)
    ds_binary.open_raster('rgb', binary_path) # for testing purpose

    fp= buzz.Footprint(
            tl=ds_rgb.rgb.fp.tl,
            size=ds_rgb.rgb.fp.size,
            rsize=ds_rgb.rgb.fp.rsize/2,
    ) #unsampling
    

    tile_size = 224
    print("Loading model")
    model_nn = FCN().load_model()
#    predicted_binary = predict_map(model_nn,tile_size,ds_rgb,fp)
#    plt.imshow(predicted_binary)
#    plt.show()
    
    print("Finding polygons")
    #test for the binary image
    binary = ds_binary.rgb.get_data(band=(1), fp=fp)
    plt.imshow(binary)
    plt.show()
#    poly = fp.find_polygons(binary)
    
    # todo : save poly
#    ds_binary.create_vector('targets', 'test.geojson', 'polygon', driver='GeoJSON')
