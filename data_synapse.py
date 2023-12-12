#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    for fp in glob.glob(os.path.join(DATA_DIR, 'synapse', f'{partition}*.npy')):
        data = np.load(fp).astype('float32')
        np.random.seed(0)
        all_data.append(data)
        print(data.shape)

    # print(len(all_data))
    return all_data

def jitter_pointcloud(pointcloud, sigma=0.5, clip=1):
    num_to_pad = 200
    pad_points = pointcloud[np.random.choice(pointcloud.shape[0], size=num_to_pad, replace=True), :]
    N, C = pad_points.shape
    pad_points += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    pointcloud = np.concatenate((pointcloud, pad_points), axis=0)
    return pointcloud

def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T

class SynapseDataset(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(SynapseDataset, self).__init__()
        self.data = load_data(partition)
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        
    def __getitem__(self, item):
        # pointcloud = self.data[item][:self.num_points]
        pointcloud1 = self.data[0][:self.num_subsampled_points].T #[:self.num_points]
        pointcloud2 = self.data[1][:self.num_subsampled_points].T

        print(pointcloud1.shape)
        print(pointcloud2.shape)
        
        # if self.subsampled:
        #     pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
        #                                                          num_subsampled_points=self.num_subsampled_points)
           
        # if self.gaussian_noise:
        #     pointcloud1 = jitter_pointcloud(pointcloud1.T).T
        #     pointcloud2 = jitter_pointcloud(pointcloud2.T).T
        
        return pointcloud1.astype('float32'), pointcloud2.astype('float32')

    def __len__(self):
        return len(self.data)
        # return self.data.shape[0]


if __name__ == '__main__':
    print('hello world')
