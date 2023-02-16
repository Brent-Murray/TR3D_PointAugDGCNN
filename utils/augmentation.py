import glob
import os
import random
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)


def rotate_points(coords):
    rotation = np.random.uniform(-180, 180)
    # Convert rotation values to radians
    rotation = np.radians(rotation)

    # Rotate point cloud
    rot_mat = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1],
        ]
    )

    aug_coords = coords
    aug_coords[:, :3] = np.matmul(aug_coords[:, :3], rot_mat)
    return aug_coords

def point_removal(coords, n, x=None):
    # Get list of ids
    idx = list(range(np.shape(coords)[0]))
    random.shuffle(idx)  # shuffle ids
    idx = np.random.choice(
        idx, n, replace=False
    )  # pick points randomly removing up to 10% of points

    # Remove random values
    aug_coords = coords[idx, :]  # remove coords
    if x is None:  # remove x
        aug_x = None
    else:
        aug_x = x[idx, :]

    return aug_coords, aug_x


def random_noise(coords, n, dim=1, x=None):
    # Random standard deviation value
    random_noise_sd = np.random.uniform(0.01, 0.025)

    # Add/Subtract noise
    if np.random.uniform(0, 1) >= 0.5:  # 50% chance to add
        aug_coords = coords + np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x + np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim
    else:  # 50% chance to subtract
        aug_coords = coords - np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x - np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim

    # Randomly choose up to 10% of augmented noise points
    use_idx = np.random.choice(aug_coords.shape[0], n, replace=False,)
    aug_coords = aug_coords[use_idx, :]  # get random points
    aug_coords = np.append(coords, aug_coords, axis=0)  # add points
    if x is None:
        aug_x = None
    else:
        aug_x = aug_x[use_idx, :]  # get random point values
        aug_x = np.append(x, aug_x, axis=0)  # add random point values # ADDED axis=0

    return aug_coords, aug_x


class AugmentPointCloudsInDF(Dataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
        self, filepath, df,
    ):
        self.filepath = filepath
        self.df = df
        super().__init__()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file path
        df_idx = self.df.iloc[idx : idx + 1]
        filename = df_idx["filename"].item()
        file = os.path.join(self.filepath, filename)

        # Read las/laz file
        coords = read_las(file, get_attributes=False)

        # Augmentation
        n = random.randint(round(len(coords)* 0.9), len(coords))
        aug_coords, x = point_removal(coords, n)
        aug_coords, x = random_noise(aug_coords, n=(len(coords)-n))
        coords = rotate_points(aug_coords)

        # Get Target
        target = df_idx.iloc[:, 10:].values.flatten().tolist()

        # To Tensor
        coords = torch.from_numpy(coords).float()
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        target = target[None, :]
        
        return coords, target
