#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The program samples original ShapeNetCore55v2 meshes into point clouds of certain sizes (random sampling).
"""

__author__ = "Daniel Koguciuk"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Daniel Koguciuk"
__email__ = "daniel.koguciuk@gmail.com"


import os
import math
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm

import pcl
import pymesh
import pyntcloud


def sample_shapenet_core_55(input_dir, output_dir, point_cloud_size):
    """
    Sample ShapeNetCore55v2 meshes into point clouds of certain sizes (random sampling).

    Args:
        input_dir (str): Input dir of ShapeNetCore55v2.
        output_dir (str): Output dir with sampled point clouds.
        point_cloud_size (int): Output size of sampling algorithm.
    """

    # Is data present?
    if not os.path.exists(input_dir):
        raise AssertionError('No ShapeNetCore55v2 dataset found in the following directory: ' + input_dir)
    
    #######################################################################
    # load taxonomy
    #######################################################################
    
    def find_class_name(taxonomy_local, class_dir_local):
        for i in range(len(taxonomy_local)):
            if class_dir_local in taxonomy_local[i]['synsetId']:
                return i, taxonomy_local[i]['name']
        return -1, 'NULL'
 
    with open(os.path.join(input_dir, 'taxonomy.json')) as data_file:
        taxonomy = json.loads(data_file.read())
    
    #######################################################################
    # load data
    #######################################################################
     
    ptclds = {}
    clsnms = []
       
    # Classes dir
    classes_dir = [os.path.join(input_dir, el) for el in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, el))]
    for idx, class_dir in enumerate(classes_dir):
           
        # Class name
        class_name = find_class_name(taxonomy, os.path.split(class_dir)[1])
        print('Converting class: ' + class_name[1])
        clsnms.append(class_name[1])
        ptclds[idx] = []
           
        # Objects dir
        objects_dir = [os.path.join(class_dir, el) for el in os.listdir(class_dir) if
                       os.path.isdir(os.path.join(class_dir, el))]
        for object_dir in tqdm(objects_dir):
               
            # Load object
            object_path = os.path.join(os.path.join(object_dir, 'models'), 'model_normalized.obj')
            pymesh_path = os.path.join(os.path.join(object_dir, 'models'), 'model_normalized_pymesh.ply')
            pclpcd_path = os.path.join(os.path.join(object_dir, 'models'),
                                       'model_normalized_' + str(point_cloud_size) + '.pcd')
             
            if not os.path.exists(object_path):
                print('No OBJ path at: ', object_path)
                continue
             
            try:
                 
                # Pymesh
                mesh = pymesh.load_mesh(object_path)
                pymesh.save_mesh(pymesh_path, mesh)
                   
                # pyntcloud
                pynt = pyntcloud.PyntCloud.from_file(pymesh_path)
                cloud = pynt.get_sample('mesh_random', n=point_cloud_size)
                cloud = cloud.values
                   
                # Zero mean
                for dim in [0, 1, 2]:
                    dim_mean = np.mean(cloud[:, dim])
                    cloud[:, dim] -= dim_mean
                   
                # Scale to unit-ball
                distances = [np.linalg.norm(point) for point in cloud]
                scale = 1. / np.max(distances)
                cloud *= scale 
                   
                # PCD
                pcd = pcl.PointCloud(cloud)
                pcl.save(pcd, pclpcd_path)         
                   
                # Append
                ptclds[idx].append(cloud)
                 
            except ValueError:
                print("An exception occurred: " + object_path)
                exit()

    #######################################################################
    # train/test split 80/20
    #######################################################################

    train_clouds = []
    train_labels = []
    test_clouds = []
    test_labels = []
    for idx in range(len(clsnms)):
        np.random.shuffle(ptclds[idx])
        split_idx = int(0.8*len(ptclds[idx]))
        train_clouds.append(ptclds[idx][:split_idx])
        train_labels.append([idx]*len(ptclds[idx][:split_idx]))
        test_clouds.append(ptclds[idx][split_idx:])
        test_labels.append([idx]*len(ptclds[idx][split_idx:]))
    train_clouds = np.array(train_clouds)
    train_labels = np.array(train_labels)
    test_clouds = np.array(test_clouds)
    test_labels = np.array(test_labels)

    #######################################################################
    # Flat pointclouds and shuffle
    #######################################################################
    
    train_clouds = np.concatenate(train_clouds, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_clouds = np.concatenate(test_clouds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    train_shuffle_idx = np.arange(len(train_clouds))
    np.random.shuffle(train_shuffle_idx)
    train_clouds = train_clouds[train_shuffle_idx]
    train_labels = train_labels[train_shuffle_idx]
    
    test_shuffle_idx = np.arange(len(test_clouds))
    np.random.shuffle(test_shuffle_idx)
    test_clouds = test_clouds[test_shuffle_idx]
    test_labels = test_labels[test_shuffle_idx]
    
    #######################################################################
    # Flat pointclouds and shuffle
    #######################################################################
    
    file_max_length = 2048
    
    for file_idx in range(math.ceil(len(train_clouds)/file_max_length)):
        filename = 'data_train_' + str(file_idx) + '.h5'
        filepath = os.path.join(output_dir, filename)
        file = h5py.File(filepath)
        start_idx = file_max_length * file_idx
        end_idx = min(len(train_clouds), file_max_length * (file_idx + 1))
        file['data'] = train_clouds[start_idx: end_idx]
        file['label'] = train_labels[start_idx: end_idx]
        file.close()

    for file_idx in range(math.ceil(len(test_clouds)/file_max_length)):
        filename = 'data_test_' + str(file_idx) + '.h5'
        filepath = os.path.join(output_dir, filename)
        file = h5py.File(filepath)
        start_idx = file_max_length*file_idx
        end_idx = min(len(test_clouds), file_max_length * (file_idx + 1))
        file['data'] = test_clouds[start_idx: end_idx]
        file['label'] = test_labels[start_idx: end_idx]
        file.close()
        
    with open(os.path.join(output_dir, 'shape_names.txt'), 'w') as f:
        for class_name in clsnms:
            f.write(class_name + '\n')


if __name__ == "__main__":
    
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="Input dir of ShapeNetCore55.v2 directory (mesh)", type=str,
                        required=True)
    parser.add_argument("-o", "--output_dir", help="Output dir of ShapeNetCore55.v2 directory (point cloud)", type=str,
                        required=True)
    parser.add_argument("-s", "--point_cloud_size", help="The size of output point clouds", type=int, required=False,
                        default=1024)
    args = vars(parser.parse_args())
    
    if os.path.exists(args['output_dir']):
        raise ValueError('Output dir already exists')
    os.mkdir(args['output_dir'])
    
    # Prep shapenet
    sample_shapenet_core_55(args['input_dir'], args['output_dir'], args['point_cloud_size'])
