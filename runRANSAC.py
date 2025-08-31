import poselib
import numpy as np

import pycolmap
import pickle
from pathlib import Path

from ipdb import set_trace



# undistort points
def undistort_points(points, camera):
    points_undistorted = poselib.undistort_points(points, camera)
    return points_undistorted

def getErrors(pose_est, pose_gt):

    R_err = np.rad2deg( np.arccos( np.clip( ( np.trace(pose_est.R @ pose_gt.rotation.matrix().T)-1 ) /2 , -1, 1 ) ) ) # in degree
    t_err = np.linalg.norm(pose_est.t - pose_gt.translation)/np.linalg.norm(pose_gt.translation) *100 # in percentage

    return R_err, t_err

# run RANSAC
def runFisheyeP4Pf(CorrsDict, query_name, solver_name = 'P4Pfr_LM', threshold = 10):

    Corrs = CorrsDict[query_name]

    opt = {'estimate_focal_length': True, 'estimate_extra_params': False, 'max_error': threshold, 
            'minimal_solver': solver_name,
            'bundle': {'verbose': False, 'loss_scale': threshold}}
        
    camera_dict = {'model': 'SIMPLE_FISHEYE', 'width': Corrs['cam'].width, 'height': Corrs['cam'].height, 'params': [1.0, 0, 0]}

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
    out['f_err'] = abs(pnpf_f - Corrs['cam'].focal_length)

    return out

def collectResults(query_list, corrs_dir, solver_name = 'P4Pfr_LM', threshold = 10):
    results = {}
    for query_name in query_list:
        results[query_name] = runFisheyeP4Pf(corrs_dir, query_name, solver_name, threshold)
    return results

if __name__ == "__main__":

    scene_list = ["sportunifront", "parakennus_out", "main_campus"]

    scene_name = scene_list[2]

    gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")

    data_folder = gt_dirs / scene_name / "processed/hloc"
    model_dir = data_folder / "sfm_superpoint+superglue"
    log_dir = data_folder / "results.txt_logs.pkl"

    query_list_dir = gt_dirs / scene_name / "processed/data/list_query.txt"
    corrs_dir = gt_dirs / scene_name / "processed/Corrs/CorrsDict.pkl"

    # read query list
    with open(query_list_dir, 'r') as f:
        query_list = f.readlines()
    query_list = [line.strip() for line in query_list]

    # read correspondences
    with open(corrs_dir, 'rb') as f:
        corrs = pickle.load(f)

    # run RANSAC
    Errors = []
    for j in range(len(query_list)):
        results_p4pfr_lm = runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_LM')
        results_p4pfr_hc_pose = runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_HC_pose')
        # results_p4pfr_hc_depth = runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_HC_depth')

        Errors.append([results_p4pfr_lm['R_err'], results_p4pfr_lm['t_err'], results_p4pfr_lm['f_err'], results_p4pfr_lm['runtime'],
                       results_p4pfr_hc_pose['R_err'], results_p4pfr_hc_pose['t_err'], results_p4pfr_hc_pose['f_err'], results_p4pfr_hc_pose['runtime']])


    Errors = np.array(Errors)
    # write to pickle
    with open(data_folder / f'RANSACresults.pkl', 'wb') as f:
        pickle.dump(Errors, f)



