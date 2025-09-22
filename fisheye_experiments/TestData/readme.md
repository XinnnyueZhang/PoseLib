### TestData overview

This folder provides a minimal example of the data for the fisheye RANSAC experiments. It contains:

- `CorrsDict.pkl`: per-query 2D–3D correspondences and metadata
- `RANSACresults_10.0_dict.pkl`: per-query pose estimation results for multiple methods at threshold 10.0

Use `loadDataAndResults.py` as a reference on loading data and check RANSAC results.

Use `runRANSAC.py` to run RANSAC experiments, but need the [fisheye poselib](https://github.com/XinnnyueZhang/PoseLib) repo I implemented (dev branch).
Installed by `python setup.py build_ext --inplace`.

### CorrsDict.pkl

Type: Python dict keyed by `query_image_name: str`.

For each query image, the value is a dict with the following fields:

- `points2D_undistorted`: float32 array (N, 2), undistorted keypoints in the SIMPLE_FISHEYE model
- `recalibrated_focal_length`: float, gt simple focal length after recalibration (pixels) from OPENCV_FISHEYE model
- `points2D`: float32 array of shape (N, 2), raw detected keypoints in pixels
- `points3D`: float32 array of shape (N, 3), corresponding 3D points in world coords
- `imageID`: int, COLMAP image id
- `cam`: pycolmap camera object (intrinsics)
- `gtPose`: pycolmap `Rigid3d` (ground-truth camera-from-world pose)
- `Rgt`: float64 array (3, 3), ground-truth rotation matrix
- `tgt`: float64 array (3,), ground-truth translation vector
- `PnP_ret`: dict with hloc calibrated PnP return info (e.g., `num_inliers`, `cam_from_world`)
- `nInliers`: int, inlier count from hloc pnp

To run RANSAC, just use `points2D_undistorted` and `points3D` as inputs.

Example:

```python
import pickle
from pathlib import Path

Corrs_dir = Path("CorrsDict.pkl")
with open(Corrs_dir, 'rb') as f:
    CorrsDict = pickle.load(f)

query_list = list(CorrsDict.keys())
q = query_list[0]
print(q, CorrsDict[q].keys())
```

### RANSACresults_10.0_dict.pkl

Type: Python dict keyed by `query_image_name: str`, and each query result is keyed by `method_name: str`.

Method names:

- `recalibrator`, `p4pfr`, `p4pfr_lm`, `p5pfr`, `p5pfr_lm`, `p4pfr_hc_pose`, `p4pfr_hc_depth`, `p3p`, `p3p_sampling_lm`, `p3p_sampling_hc`, `p3p_given_gtf`, `p3p_given_anyCalibf1`, `p3p_given_anyCalibf4`

Each method entry is a dict with:

- `runtime`: float, runtime in milliseconds
- `R_err`: float, rotation error in degrees versus `gtPose`
- `t_err`: float, translation error in centimeters versus `gtPose`
- `f_err`: float, absolute focal length error in pixels (0 for known-f cases)
- `quat`: float64 array (4,), estimated orientation quaternion
- `t`: float64 array (3,), estimated translation (world to camera)
- `R`: float64 array (3, 3), estimated rotation matrix
- `f`: float, estimated focal length (pixels) if applicable
- `info`: dict with RANSAC internals, e.g., `iterations`, `num_inliers`, `inliers`
Example:

```python
import pickle
from pathlib import Path

results_path = Path("RANSACresults_10.0_dict.pkl")
with open(results_path, 'rb') as f:
    Results = pickle.load(f)

query_result = list(Results.values())[0]
methods = list(query_result.keys())
print(methods[0])
print('focal length error:', query_result[methods[0]]['f_err'], 'px')
print('runtime:', query_result[methods[0]]['runtime'], 'ms')
```

### Notes

- Threshold for RANSAC is 10 pxiels. And the same threshold is used for bundle.

