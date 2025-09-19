import poselib
import numpy as np
import pandas as pd
import pycolmap
import pickle
from pathlib import Path
import argparse
from ipdb import set_trace

from tqdm import tqdm


def getErrors(pose_est, pose_gt):
    """Calculate rotation and translation errors between estimated and ground truth poses."""
    R_err = np.rad2deg(np.arccos(np.clip((np.trace(pose_est.R @ pose_gt.rotation.matrix().T) - 1) / 2, -1, 1)))
    t_err = np.linalg.norm(pose_est.t - pose_gt.translation) * 100  # in cm
    return R_err, t_err


def extract_error_metrics(result):
    """Extract error metrics from a single result dictionary."""
    return [
        result['R_err'], result['t_err'], result['f_err'], 
        result['runtime'], result['info']['iterations'], result['info']['num_inliers']
    ]


def create_pandas_entry(query_name, method, result):
    """Create a pandas entry for a single result."""
    return {
        'query': query_name,
        'method': method,
        'R Error': result['R_err'],
        't Error': result['t_err'],
        'f Error': result['f_err'],
        'runtime': result['runtime'],
        'iterations': result['info']['iterations'],
        'num_inliers': result['info']['num_inliers'],
        'inliers': result['info']['inliers']
    }

# run calibrated RANSAC 
def runp3p(CorrsDict, query_name, threshold = 10):

    Corrs = CorrsDict[query_name]

    simplefisheye_cam = poselib.Camera('SIMPLE_FISHEYE', [Corrs['recalibrated_focal_length'], 0, 0], Corrs['cam'].width, Corrs['cam'].height)
    points2D_unprojected = simplefisheye_cam.unproject(Corrs['points2D_undistorted'])
    points2D_unprojected = np.array(points2D_unprojected)

    points2D_unprojected = points2D_unprojected[:,:2]/points2D_unprojected[:,2:3]


    opt = {'estimate_focal_length': False, 'estimate_extra_params': False, 'max_error': threshold, 
           'ransac': {'min_iterations': 100, 'max_iterations': 2000},
            'bundle': {'verbose': False, 'loss_scale': threshold}}
    camera_dict = {'model': 'SIMPLE_PINHOLE', 'width': Corrs['cam'].width, 'height': Corrs['cam'].height, 'params': [1.0, 0, 0]}
    pose, info = poselib.estimate_absolute_pose(points2D_unprojected, Corrs['points3D'], camera_dict, opt)

    pnpf_R, pnpf_q, pnpf_t = pose.pose.R, pose.pose.q, pose.pose.t

    out = {}
    out['quat'] = pnpf_q
    out['t'] = pnpf_t
    out['f'] = Corrs['recalibrated_focal_length'] # gt focal length used for normalize the points
    out['R'] = pnpf_R
    out['info'] = info
    out['runtime'] = info['runtime']/1e6 # in ms recording time in Python

    R_err, t_err = getErrors(pose.pose, Corrs['gtPose'])
    out['R_err'] = R_err
    out['t_err'] = t_err
    out['f_err'] = 0.0

    return out

def runRecalibratedRANSAC(CorrsDict, query_name, threshold = 10):

    Corrs = CorrsDict[query_name]

    opt = {'estimate_focal_length': True, 'estimate_extra_params': True, 'max_error': threshold, 
            'ransac': {'min_iterations': 10, 'max_iterations': 2000},
            'bundle': {'verbose': False, 'loss_scale': threshold}}
            
    camera_dict = {'model': 'SIMPLE_FISHEYE', 'width': Corrs['cam'].width, 'height': Corrs['cam'].height, 
        'params': [1.0, 0, 0]}

    pose, info = poselib.estimate_absolute_pose(Corrs['points2D_undistorted'], Corrs['points3D'], camera_dict, opt)
    pnpf_R, pnpf_q, pnpf_t, pnpf_f = pose.pose.R, pose.pose.q, pose.pose.t, pose.camera.focal()

    out = {}
    out['quat'] = pnpf_q
    out['t'] = pnpf_t
    out['f'] = Corrs['recalibrated_focal_length'] # gt focal length used for normalize the points
    out['R'] = pnpf_R
    out['info'] = info
    out['runtime'] = info['runtime']/1e6 # in ms recording time in Python

    R_err, t_err = getErrors(pose.pose, Corrs['gtPose'])
    out['R_err'] = R_err
    out['t_err'] = t_err
    out['f_err'] = abs(pnpf_f - Corrs['recalibrated_focal_length'])

    return out

# run RANSAC
def runFisheyeP4Pf(CorrsDict, query_name, solver_name = 'P4Pfr_LM', threshold = 10, predictedfDict = None, anycalib_model = 'simple_kb:1'):

    Corrs = CorrsDict[query_name]

    opt = {'estimate_focal_length': True, 'estimate_extra_params': False, 'max_error': threshold, 
            'minimal_solver': solver_name, 'ransac': {'min_iterations': 10, 'max_iterations': 2000},
            'bundle': {'verbose': False, 'loss_scale': threshold}}
    if solver_name == 'P3P_givenf' and predictedfDict is not None:
        predicted_focal_length = predictedfDict[query_name][anycalib_model]
        focal_intiial = predicted_focal_length
    else:
        focal_intiial = Corrs['recalibrated_focal_length']
    camera_dict = {'model': 'SIMPLE_FISHEYE', 'width': Corrs['cam'].width, 'height': Corrs['cam'].height, 
        'params': [focal_intiial, 0, 0]}

    pose, info = poselib.estimate_absolute_pose_fisheye(Corrs['points2D_undistorted'], Corrs['points3D'], camera_dict, opt)

    pnpf_R, pnpf_q, pnpf_t, pnpf_f = pose.pose.R, pose.pose.q, pose.pose.t, pose.camera.focal()

    out = {}
    out['quat'] = pnpf_q
    out['t'] = pnpf_t
    out['f'] = pnpf_f
    out['R'] = pnpf_R
    out['info'] = info
    out['runtime'] = info['runtime']/1e6 # in ms

    R_err, t_err = getErrors(pose.pose, Corrs['gtPose'])
    out['R_err'] = R_err
    out['t_err'] = t_err
    out['f_err'] = abs(pnpf_f - Corrs['recalibrated_focal_length'])

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--process_name", type=str, default="covisible80")
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()

    scene_list = ["festia_out_corridor", "sportunifront", "parakennus_out", "main_campus", "Kitchen_In",
        "meetingroom", "night_out", "outcorridor", "parakennus"]
        #  , "upstairs"

    # scene_name = scene_list[0]
    for scene_name in scene_list:
        print("running RANSAC for scene: ", scene_name)

        gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")

        data_folder = gt_dirs / scene_name / f"processed_{args.process_name}/hloc"

        # results_file = data_folder / f'RANSACresults_{args.process_name}.pkl'
        results_file = data_folder / f'RANSACresults_{args.threshold}.pkl'

        # if results_file.exists():
        #     continue

        model_dir = data_folder / "sfm_superpoint+superglue"
        log_dir = data_folder / "results.txt_logs.pkl"

        query_list_dir = gt_dirs / scene_name / f"processed_{args.process_name}/data/list_query.txt"
        corrs_dir = gt_dirs / scene_name / f"processed_{args.process_name}/Corrs/CorrsDict.pkl"

        ## read query list
        with open(query_list_dir, 'r') as f:
            query_list = f.readlines()
        query_list = [line.strip() for line in query_list]

        ## read correspondences
        with open(corrs_dir, 'rb') as f:
            corrs = pickle.load(f)
        
        # load predicted focal length
        ## load anycalib
        anycalib_dir = gt_dirs / scene_name / "anyCalib_focal.pkl"
        with open(anycalib_dir, 'rb') as f:
            anycalib = pickle.load(f)

        # Define solver configurations
        solver_configs = [
            ('recalibrator', lambda: runRecalibratedRANSAC(corrs, query_list[j], threshold = args.threshold)),
            ('p4pfr', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr', threshold = args.threshold)),
            ('p4pfr_lm', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_LM', threshold = args.threshold)),
            ('p5pfr', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P5Pfr', threshold = args.threshold)),
            ('p5pfr_lm', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P5Pfr_LM', threshold = args.threshold)),
            ('p4pfr_hc_pose', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_HC_pose', threshold = args.threshold)),
            ('p4pfr_hc_depth', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_HC_depth', threshold = args.threshold)),
            ('p3p', lambda: runp3p(corrs, query_list[j], threshold = args.threshold)),
            ('p3p_sampling_lm', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P3P_sampling_LM', threshold = args.threshold)),
            ('p3p_sampling_hc', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P3P_sampling_HC', threshold = args.threshold)),
            ('p3p_given_gtf', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P3P_givenf', threshold = args.threshold)),
            ('p3p_given_anyCalibf1', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P3P_givenf', predictedfDict = anycalib, anycalib_model = 'simple_kb:1', threshold = args.threshold)),
            ('p3p_given_anyCalibf4', lambda: runFisheyeP4Pf(corrs, query_list[j], 'P3P_givenf', predictedfDict = anycalib, anycalib_model = 'simple_kb:4', threshold = args.threshold)),
        ]

        # run RANSAC
        Errors_pd = []
        Errors_np = {}
        Errors_dict = {}
        for j in tqdm(range(len(query_list))):
            # Run all solvers and collect results
            results = {}
            for method_name, solver_func in solver_configs:
                results[method_name] = solver_func()

            # Extract error metrics for numpy array
            error_array = []
            for method_name, result in results.items():
                error_array.extend(extract_error_metrics(result))
            Errors_np[query_list[j]] = np.array(error_array)

            # Store detailed results
            Errors_dict[query_list[j]] = results

            # Create pandas entries
            for method_name, result in results.items():
                Errors_pd.append(create_pandas_entry(query_list[j], method_name, result))

        Errors_pd = pd.DataFrame(Errors_pd)

        # write to pickle
        with open(results_file, 'wb') as f:
            pickle.dump(Errors_np, f)

        with open(results_file.parent / f'RANSACresults_pd.pkl', 'wb') as f:
            pickle.dump(Errors_pd, f)

        with open(results_file.parent / f'RANSACresults_dict.pkl', 'wb') as f:
            pickle.dump(Errors_dict, f)
