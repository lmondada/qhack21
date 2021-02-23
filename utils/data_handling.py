
"""
    These functions are used to load data from the ESA dataset
    and to format it into convenient formats.

    We use pandas DataFrame and Series to load and transform the data.
"""

import pandas as pd
import numpy as np

import os
from pathlib import Path
import urllib.request
import zipfile

from .utils import print_progress


def download_data():
    ZIP_FILE = Path("data.zip")
    TRAIN_ZIP_FILE = Path("data/kelvins_competition_data/train_data.zip")
    FOLDER_NAME = "Collision Avoidance Challenge - Dataset"
    URL = "https://zenodo.org/record/4463683/files/Collision%20Avoidance%20Challenge%20-%20Dataset.zip"
    
    with print_progress("Downloading data"):
        with urllib.request.urlopen(URL) as url_f:
            with open(ZIP_FILE, 'wb') as out_f:
                out_f.write(url_f.read())

    with print_progress(f'Unzipping {ZIP_FILE}'):
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_f:
            zip_f.extractall('.')

    os.rename(FOLDER_NAME, 'data')
    os.remove(ZIP_FILE)

    with print_progress(f'Unzipping {TRAIN_ZIP_FILE}'):
        with zipfile.ZipFile(TRAIN_ZIP_FILE, 'r') as zip_f:
            zip_f.extractall(os.path.split(TRAIN_ZIP_FILE)[0])
    os.remove(TRAIN_ZIP_FILE)
    
    print("Done. Data available at data/")

DATA_FILES = {
    "train_data": "./data/kelvins_competition_data/train_data.csv",
    "test_data": "./data/kelvins_competition_data/test_data.csv",
}

def load_data(data_type="train_data"):
    """ Load data from file into DataFrame
    
    Returns data in a pandas dataframe. This downloads data from remote if necessary
    and stores it in `data/` folder.

    The optional argument should be "train_data" or "test_data" and indicates whether the
    training or the testing data should be loaded.
    """
    filename = DATA_FILES[data_type]
    if not os.path.exists('./data'):
        download_data()
    with print_progress('Reading data'):
        return pd.read_csv(filename)

# ESA schema to describe RTN matrix and vectors
_cov_structure = [
    ["x_sigma_r", "x_ct_r", "x_cn_r", "x_crdot_r", "x_ctdot_r", "x_cndot_r"],
    ["x_ct_r", "x_sigma_t", "x_cn_t", "x_crdot_t", "x_ctdot_t", "x_cndot_t"],
    ["x_cn_r", "x_cn_t", "x_sigma_n", "x_crdot_n", "x_ctdot_n", "x_cndot_n"],
    ["x_crdot_r", "x_crdot_t", "x_crdot_n", "x_sigma_rdot", "x_ctdot_rdot", "x_cndot_rdot"],
    ["x_ctdot_r", "x_ctdot_t", "x_ctdot_n", "x_ctdot_rdot", "x_sigma_tdot", "x_cndot_tdot"],
    ["x_cndot_r", "x_cndot_t", "x_cndot_n", "x_cndot_rdot", "x_cndot_tdot", "x_sigma_ndot"],
]
_vec_structure = [
    "relative_x_r", "relative_x_t", "relative_x_n",
]
_sigma_structure = [
    "x_sigma_r", "x_sigma_t", "x_sigma_n", "x_sigma_rdot", "x_sigma_tdot", "x_sigma_ndot"
]

def _get_matrix(x, data):
    """ Transforms multi-column input into a numpy 2dim array form """
    to_sigma = lambda row_data: np.array([
        row_data[name.replace("x_", x + "_")] for name in _sigma_structure
    ])
    to_matrix = lambda row_data: np.array([
        [row_data[name.replace("x_", x + "_")] if not name.find("sigma") >= 0 else 1. for name in row_names]
        for row_names in _cov_structure
    ]) * to_sigma(row_data).reshape(-1, 1) * to_sigma(row_data).reshape(1, -1)
    ret = data.apply(
        func=to_matrix,
        axis="columns",
        result_type="reduce",
    )
    return ret

def _get_relative_vector(vec_type, data) -> pd.Series:
    """ Transforms multi-column input into a numpy 1dim array form """
    vec_type = "_" + vec_type + "_"
    to_vec = lambda row_data: np.array([
        row_data[name.replace("_x_", vec_type)] for name in _vec_structure
    ])
    ret = data.apply(
        func=to_vec,
        axis="columns",
        result_type="reduce",
    )
    return ret

def get_cmatrix(data) -> pd.Series:
    """ Get chaser matrix from raw ESA data (as a DataFrame) into numpy array form """
    return _get_matrix("c", data)

def get_tmatrix(data) -> pd.Series:
    """ Get chaser matrix from raw ESA data (as a DataFrame) into numpy array form """
    return _get_matrix("t", data)

def c2t_matrix(rel_vel: np.ndarray) -> np.ndarray:
    """ Basis change matrix from chaser frame to target frame. """
    c_t = rel_vel
    c_t = c_t / np.linalg.norm(c_t)
    c_r = np.array([0, 0, 1]) # approximate radial for chaser = radial for target
    c_n = np.cross(c_t, c_r)
    c_n = c_n / np.linalg.norm(c_n)
    c_r = np.cross(c_n, c_t)
    c_r = c_r / np.linalg.norm(c_r)
    return np.array([[c_n[0], c_t[0], c_r[0]],
                     [c_n[1], c_t[1], c_r[1]],
                     [c_n[2], c_t[2], c_r[2]]])

def get_relpos(data) -> pd.Series:
    """ Get relative position from raw ESA data into numpy array form """
    return _get_relative_vector("position", data)

def get_relvel(data) -> pd.Series:
    """ Get relative velocity from raw ESA data into numpy array form """
    return _get_relative_vector("velocity", data)

def get_combined_cov_pos(data) -> pd.Series:
    """
    Get combined position covariance matrix of chaser and target objects

    Returns the 3x3 position covariance matrices of the dataset, expressed as the combined
    uncertainty of both chaser and target objects.
    We discard velocity terms and only consider position.

    Note: This is what we want to encode in our quantum circuit!
    """
    # TODO this should be vectorised really
    out = []
    for vel, c_cov, t_cov in zip(get_relvel(data), get_cmatrix(data), get_tmatrix(data)):
        c_poscov = c_cov[:3, :3]
        t_poscov = t_cov[:3, :3]
        c2t_mat = c2t_matrix(vel)
        out.append(c2t_mat @ c_poscov @ c2t_mat.T + t_poscov)
    return pd.Series(out, index=data.index)