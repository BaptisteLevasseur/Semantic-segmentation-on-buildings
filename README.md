# Semantic segmentation on buildings

### Project 

This project is based on the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).
The aim is to build an algorithm based on neural network performing semantic segmentation on buildings.

The open source library [buzzard](https://github.com/airware/buzzard/) developped by Airware has been used to manage georeferenced data. 
The results are saved and store at the geoJSON format.

### Main ideas

The solution implemented is inspired by the [TernausNet](https://arxiv.org/abs/1801.05746). It is a autoencoder that reuses the features learned during the encoding part for the decoding part.
The used architecture was :

![alt text](Results/model_display.png)

### Resuts

