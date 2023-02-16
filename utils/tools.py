import glob
import os
from datetime import datetime
from pathlib import Path
from itertools import cycle, islice

import laspy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from plyer import notification
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


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
    
    
class PointCloudsInDF(Dataset):
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

        # Get Target
        target = df_idx.iloc[:, 10:].values.flatten().tolist()

        # To Tensor
        coords = torch.from_numpy(coords).float()
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        target = target[None, :]

        return coords, target


def write_las(outpoints, outfilepath, attribute_dict={}):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[0]
    las.y = outpoints[1]
    las.z = outpoints[2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)
    
class IOStream:
    # Adapted from https://github.com/vinits5/learning3d/blob/master/examples/train_pointnet.py
    def __init__(self, path):
        # Open file in append
        self.f = open(path, "a")

    def cprint(self, text):
        # Print and write text to file
        print(text)  # print text
        self.f.write(text + "\n")  # write text and new line
        self.f.flush  # flush file

    def close(self):
        self.f.close()  # close file


def _init_(model_name):
    # Create folder structure
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + model_name):
        os.makedirs("checkpoints/" + model_name)
    if not os.path.exists("checkpoints/" + model_name + "/models"):
        os.makedirs("checkpoints/" + model_name + "/models")
    if not os.path.exists("checkpoints/" + model_name + "/output"):
        os.makedirs("checkpoints/" + model_name + "/output")
    if not os.path.exists("checkpoints/" + model_name + "/output/laz"):
        os.makedirs("checkpoints/" + model_name + "/output/laz")


def make_confusion_matrix(
    cm,
    labels,
    normalize=False,
    accuracy=None,
    precision=None,
    recall=None,
    f1=None,
    figsize=None,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = ""

    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        mask=~off_diag_mask,
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="black",
        xticklabels=labels,
        yticklabels=labels,
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        mask=off_diag_mask,
        cmap="Reds",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )

    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        accuracy, precision, recall, f1
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label" + stats_text)


def delete_files(root_dir, glob="*"):
    # List files in root_dir with glob
    files = list(Path(root_dir).glob(glob))

    # Delete files
    for f in files:
        os.remove(f)


def check_multi_gpu(n_gpus):
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1 and n_gpus > 1:
        multi_gpu = True
        print("Using Multiple GPUs")
    else:
        multi_gpu = False
        
    return multi_gpu


def notifi(title, message):
    # Creates a pop-up notification
    notification.notify(title=title, message=message, timeout=10)
