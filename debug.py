from pathlib import Path

from hloc import visualization

import pycolmap
import numpy as np
import pickle
import pandas as pd
from hloc.utils.io import read_image
from hloc.utils.viz import plot_images

from ipdb import set_trace

scene = "parakennus_out"
gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")
images = gt_dirs / scene / "images"

outputs = gt_dirs / scene / "processed_remove_neighbours/hloc"

sfm_dir = outputs / "sfm_superpoint+superglue"

# model = gt_dirs / scene / "colmap/model"

results = outputs / "results.txt"

plot_images([read_image(images / 'cam2/IMG_20250330_115955_00_302_fisheye2.jpg')])
