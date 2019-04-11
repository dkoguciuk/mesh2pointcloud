#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The program samples original ModelNet40 meshes into point clouds of certain sizes (random sampling).
"""

__author__ = "Daniel Koguciuk"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Daniel Koguciuk"
__email__ = "daniel.koguciuk@gmail.com"

import os
import sys
import math
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import pcl
import pymesh
import pyntcloud

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'input'))
sys.path.append(os.path.join(BASE_DIR, 'output'))


def sample_modelnet_40(input_dir, output_dir, point_cloud_size):
    """
    Sample ModelNet40 meshes into point clouds of certain sizes (random sampling).

    Args:
        input_dir (str): Input dir of ModelNet40.
        output_dir (str): Output dir with sampled point clouds.
        point_cloud_size (int): Output size of sampling algorithm.
    """

    # Is data present?
    if not os.path.exists(input_dir):
        raise AssertionError('No ModelNet40 dataset found in the following directory: ' + input_dir)

    #######################################################################
    # load data
    #######################################################################

    train_clouds = []
    train_labels = []
    test_clouds = []
    test_labels = []
    clsnms = []

    # Classes dir
    class_names = [el for el in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, el))]
    class_dirs = [os.path.join(input_dir, el) for el in class_names]

    for idx, class_dir in enumerate(class_dirs):

        # Class name
        class_name = os.path.split(class_dir)[1]
        clsnms.append(class_name)

        # For train/test split
        for part_name in ['train', 'test']:

            # For each file in part
            print('Converting', part_name, 'part of', class_name, 'class')
            part_dir = os.path.join(class_dir, part_name)
            for filename in tqdm([f for f in os.listdir(part_dir) if '.off' in f]):

                # Load object
                f = filename.split('.')
                f_ply = str(f[0]) + '.ply'
                f_pcd = str(f[0]) + '_' + str(point_cloud_size) + '_' + '.pcd'
                object_path = os.path.join(os.path.join(part_dir, filename))
                pymesh_path = os.path.join(os.path.join(part_dir, f_ply))
                pclpcd_path = os.path.join(os.path.join(part_dir, f_pcd))

                # No off
                if not os.path.exists(object_path):
                    print('No OFF path at: ', object_path)
                    continue

                try:

                    # OFF file fix
                    need_fix = False
                    with open(object_path) as f:
                        first_line = f.readline()
                        if 'OFF' in first_line and len(first_line) != 4:
                            need_fix = True
                            content = f.readlines()
                    if need_fix:
                        with open(object_path, 'w') as f:
                            f.write(first_line[:3] + '\n')
                            f.write(first_line[3:])
                            for c in content:
                                f.write(c)

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
                    if part_name == 'train':
                        train_clouds.append(cloud)
                        train_labels.append(idx)
                    else:
                        test_clouds.append(cloud)
                        test_labels.append(idx)

                except ValueError:
                    print("An exception occurred: " + object_path)
                    return
 
    # Numpy arr
    train_clouds = np.array(train_clouds)
    train_labels = np.array(train_labels)
    test_clouds = np.array(test_clouds)
    test_labels = np.array(test_labels)
    clsnms = np.array(clsnms)

    #######################################################################
    # Flat pointclouds and shuffle
    #######################################################################

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

    for file_idx in range(math.ceil(len(train_clouds) / file_max_length)):
        filename = 'data_train_' + str(file_idx) + '.h5'
        filepath = os.path.join(output_dir, filename)
        file = h5py.File(filepath)
        start_idx = file_max_length * file_idx
        end_idx = min(len(train_clouds), file_max_length * (file_idx + 1))
        file['data'] = train_clouds[start_idx: end_idx]
        file['label'] = train_labels[start_idx: end_idx]
        file.close()

    for file_idx in range(math.ceil(len(test_clouds) / file_max_length)):
        filename = 'data_test_' + str(file_idx) + '.h5'
        filepath = os.path.join(output_dir, filename)
        file = h5py.File(filepath)
        start_idx = file_max_length * file_idx
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
    parser.add_argument("-i", "--input_dir", help="Input dir of ModelNet40 directory (mesh)", type=str,
                        required=True)
    parser.add_argument("-o", "--output_dir", help="Output dir of ModelNet40 directory (point cloud)", type=str,
                        required=True)
    parser.add_argument("-s", "--point_cloud_size", help="The size of output point clouds", type=int, required=False,
                        default=1024)
    args = vars(parser.parse_args())

    if os.path.exists(args['output_dir']):
        raise ValueError('Output dir already exists')
    os.mkdir(args['output_dir'])

    # Prep modelnet40
    sample_modelnet_40(args['input_dir'], args['output_dir'], args['point_cloud_size'])
