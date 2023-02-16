import glob
import os
import math
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


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


def farthest_point_sampling(coords, k, attrs=None):
    # Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html

    # Get points into numpy array
    points = np.array(coords)

    # Get points index values
    idx = np.arange(len(coords))

    # Initialize use_idx
    use_idx = np.zeros(k, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of k samples
    for i in range(1, k):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)

    coords = coords[use_idx]
    if attrs:
        for key, vals in attrs.items():
            attrs[key] = vals[use_idx]
        return coords, attrs
    else:
        return coords


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

    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)


def get_cluster_distribution(coords, clusters):
    clust_dists = []
    for i in range(len(np.unique(clusters))):
        coords_0 = coords[clusters == i]
        clust_dist = len(coords_0) / len(coords)
        clust_dists.append(clust_dist)

    return clust_dists


def check_distribution(dists, k):
    if sum(dists) != k:
        y = sum(dists) - k
        dists[np.argmax(dists)] = dists[np.argmax(dists)] - y

    return dists


def cluster_fps(coords, clusters, k):
    clust_dists = get_cluster_distribution(coords, clusters)
    clust_dists = [round(k * i) for i in clust_dists]
    clust_dists = check_distribution(clust_dists, k)

    idx_list = []
    for i in np.unique(clusters):
        coords_cluster = coords[clusters == i]
        if clust_dists[i] != 0:
            coords_fps = farthest_point_sampling(coords_cluster, clust_dists[i])
        else:
            pass
        idx_list.append(coords_fps)
    coords = np.concatenate(idx_list, axis=0)

    return coords
    
    
def resample_point_clouds(root_dir, max_points_list, samp_meth, glob="*.las", bins=None, clusters=None):
    # Create training set for each point density
    files = list(Path(root_dir).glob(glob))

    for max_points in tqdm(max_points_list, desc="Total: ", leave=False, colour="blue"):
        # Make folders
        if not os.path.exists(os.path.join(root_dir, "datasets")):
            os.makedirs(os.path.join(root_dir, "datasets"))
        if not os.path.exists(
            os.path.join(root_dir, "datasets", samp_meth)
        ):
            os.makedirs(os.path.join(root_dir, "datasets", samp_meth))
        if not os.path.exists(
            os.path.join(root_dir, "datasets", samp_meth, str(max_points))
        ):
            os.makedirs(
                os.path.join(
                    root_dir, "datasets", samp_meth, str(max_points)
                )
            )

        for file in tqdm(
            files, desc="Max Points: " + str(max_points), leave=False, colour="red"
        ):
            # Read las/laz file
            coords, attrs = read_las(file, get_attributes=True)
            filename = str(file).split("\\")[-1]

            # Resample number of points to max_points
            if coords.shape[0] >= max_points:
                if samp_meth == "random":
                    use_idx = np.random.choice(
                        coords.shape[0], max_points, replace=False
                    )
                if samp_meth == "fps":
                    coords = farthest_point_sampling(coords, max_points)
                if samp_meth == "height_fps":
                    use_idx = height_fps(coords, max_points, bins)
                if samp_meth == "cluster_fps":
                    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(coords)
                    coords = cluster_fps(coords, kmeans.labels_, max_points)
            else:
                use_idx = np.random.choice(coords.shape[0], max_points, replace=True)
                coords = coords[use_idx, :]
            

            # Write out files
            write_las(
                coords,
                os.path.join(
                    root_dir,
                    "datasets",
                    samp_meth,
                    str(max_points),
                    filename,
                ),
            )
