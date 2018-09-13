# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:30:22 2018

@author: HP
"""
import geometry_store
import geometry

shapelyGeometries = [
    geometry.Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]),
    geometry.Polygon([(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)]),
]

path = 'test.shp'
geometry_store.save(path, geometry_store.proj4LL, shapelyGeometries)
result = geometry_store.load(path)
