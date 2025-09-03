import poselib
import numpy as np
import pandas as pd
import pycolmap
import pickle
from pathlib import Path

from ipdb import set_trace

from tqdm import tqdm


def getErrors(pose_est, pose_gt):

    R_err = np.rad2deg( np.arccos( np.clip( ( np.trace(pose_est.R @ pose_gt.rotation.matrix().T)-1 ) /2 , -1, 1 ) ) ) # in degree
    # t_err = np.linalg.norm(pose_est.t - pose_gt.translation)/np.linalg.norm(pose_gt.translation) *100 # in percentage
    t_err = np.linalg.norm(pose_est.t - pose_gt.translation)*100 # in mm

    return R_err, t_err

# run calibrated RANSAC 
def runp3p(CorrsDict, query_name, threshold = 10):

    Corrs = CorrsDict[query_name]

    simplefisheye_cam = poselib.Camera('SIMPLE_FISHEYE', [Corrs['cam'].focal_length, 0, 0], Corrs['cam'].width, Corrs['cam'].height)
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
    out['f'] = Corrs['cam'].focal_length # gt focal length used for normalize the points
    out['R'] = pnpf_R
    out['info'] = info
    out['runtime'] = info['runtime']/1e6 # in ms recording time in Python

    R_err, t_err = getErrors(pose.pose, Corrs['gtPose'])
    out['R_err'] = R_err
    out['t_err'] = t_err
    out['f_err'] = 0.0

    return out

# run RANSAC
def runFisheyeP4Pf(CorrsDict, query_name, solver_name = 'P4Pfr_LM', threshold = 10):

    Corrs = CorrsDict[query_name]

    opt = {'estimate_focal_length': True, 'estimate_extra_params': False, 'max_error': threshold, 
            'minimal_solver': solver_name, 'ransac': {'min_iterations': 100, 'max_iterations': 2000},
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
    out['f_err'] = abs(pnpf_f - Corrs['recalibrated_focal_length'])

    return out

def collectResults(query_list, corrs_dir, solver_name = 'P4Pfr_LM', threshold = 10):
    results = {}
    for query_name in query_list:
        results[query_name] = runFisheyeP4Pf(corrs_dir, query_name, solver_name, threshold)
    return results

if __name__ == "__main__":

    scene_list = ["festia_out_corridor", "sportunifront", "parakennus_out", "main_campus", "Kitchen_In", "meetingroom", "night_out", "outcorridor", "parakennus", "upstairs"]

    # scene_name = scene_list[0]
    for scene_name in scene_list:
        print("running RANSAC for scene: ", scene_name)

        gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")

        data_folder = gt_dirs / scene_name / "processed_remove_neighbours/hloc"

        results_file = data_folder / f'RANSACresults_remove_neighbours.pkl'

        # if results_file.exists():
        #     continue

        model_dir = data_folder / "sfm_superpoint+superglue"
        log_dir = data_folder / "results.txt_logs.pkl"

        query_list_dir = gt_dirs / scene_name / "processed_remove_neighbours/data/list_query.txt"
        corrs_dir = gt_dirs / scene_name / "processed_remove_neighbours/Corrs/CorrsDict.pkl"

        # read query list
        with open(query_list_dir, 'r') as f:
            query_list = f.readlines()
        query_list = [line.strip() for line in query_list]

        # read correspondences
        with open(corrs_dir, 'rb') as f:
            corrs = pickle.load(f)

        # run RANSAC
        errors = []
        Errors_np = {}
        for j in tqdm(range(len(query_list))):

            results_p3p = runp3p(corrs, query_list[j])
            results_p4pfr_lm = runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_LM')
            results_p4pfr_hc_pose = runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_HC_pose')
            results_p4pfr_hc_depth = runFisheyeP4Pf(corrs, query_list[j], 'P4Pfr_HC_depth')
            results_p3p_sampling_lm = runFisheyeP4Pf(corrs, query_list[j], 'P3P_sampling_LM')
            results_p3p_sampling_hc = runFisheyeP4Pf(corrs, query_list[j], 'P3P_sampling_HC')

            error_ = [results_p3p['R_err'], results_p3p['t_err'], results_p3p['f_err'], results_p3p['runtime'], results_p3p['info']['iterations'], results_p3p['info']['num_inliers'],
                results_p4pfr_lm['R_err'], results_p4pfr_lm['t_err'], results_p4pfr_lm['f_err'], results_p4pfr_lm['runtime'], results_p4pfr_lm['info']['iterations'], results_p4pfr_lm['info']['num_inliers'],
                results_p4pfr_hc_pose['R_err'], results_p4pfr_hc_pose['t_err'], results_p4pfr_hc_pose['f_err'], results_p4pfr_hc_pose['runtime'], results_p4pfr_hc_pose['info']['iterations'], results_p4pfr_hc_pose['info']['num_inliers'],
                results_p4pfr_hc_depth['R_err'], results_p4pfr_hc_depth['t_err'], results_p4pfr_hc_depth['f_err'], results_p4pfr_hc_depth['runtime'], results_p4pfr_hc_depth['info']['iterations'], results_p4pfr_hc_depth['info']['num_inliers'],
                results_p3p_sampling_lm['R_err'], results_p3p_sampling_lm['t_err'], results_p3p_sampling_lm['f_err'], results_p3p_sampling_lm['runtime'], results_p3p_sampling_lm['info']['iterations'], results_p3p_sampling_lm['info']['num_inliers'],
                results_p3p_sampling_hc['R_err'], results_p3p_sampling_hc['t_err'], results_p3p_sampling_hc['f_err'], results_p3p_sampling_hc['runtime'], results_p3p_sampling_hc['info']['iterations'], results_p3p_sampling_hc['info']['num_inliers']]
            Errors_np[query_list[j]] = np.array(error_)

            errors.append({'query': query_list[j], 'method': 'p3p', 'R Error': results_p3p['R_err'], 't Error': results_p3p['t_err'], 'f Error': results_p3p['f_err'], 'runtime': results_p3p['runtime'], 'iterations': results_p3p['info']['iterations'], 'num_inliers': results_p3p['info']['num_inliers']})
            errors.append({'query': query_list[j], 'method': 'p4pfr_lm', 'R Error': results_p4pfr_lm['R_err'], 't Error': results_p4pfr_lm['t_err'], 'f Error': results_p4pfr_lm['f_err'], 'runtime': results_p4pfr_lm['runtime'], 'iterations': results_p4pfr_lm['info']['iterations'], 'num_inliers': results_p4pfr_lm['info']['num_inliers']})
            errors.append({'query': query_list[j], 'method': 'p4pfr_hc_pose', 'R Error': results_p4pfr_hc_pose['R_err'], 't Error': results_p4pfr_hc_pose['t_err'], 'f Error': results_p4pfr_hc_pose['f_err'], 'runtime': results_p4pfr_hc_pose['runtime'], 'iterations': results_p4pfr_hc_pose['info']['iterations'], 'num_inliers': results_p4pfr_hc_pose['info']['num_inliers']})
            errors.append({'query': query_list[j], 'method': 'p4pfr_hc_depth', 'R Error': results_p4pfr_hc_depth['R_err'], 't Error': results_p4pfr_hc_depth['t_err'], 'f Error': results_p4pfr_hc_depth['f_err'], 'runtime': results_p4pfr_hc_depth['runtime'], 'iterations': results_p4pfr_hc_depth['info']['iterations'], 'num_inliers': results_p4pfr_hc_depth['info']['num_inliers']})
            errors.append({'query': query_list[j], 'method': 'p3p_sampling_lm', 'R Error': results_p3p_sampling_lm['R_err'], 't Error': results_p3p_sampling_lm['t_err'], 'f Error': results_p3p_sampling_lm['f_err'], 'runtime': results_p3p_sampling_lm['runtime'], 'iterations': results_p3p_sampling_lm['info']['iterations'], 'num_inliers': results_p3p_sampling_lm['info']['num_inliers']})
            errors.append({'query': query_list[j], 'method': 'p3p_sampling_hc', 'R Error': results_p3p_sampling_hc['R_err'], 't Error': results_p3p_sampling_hc['t_err'], 'f Error': results_p3p_sampling_hc['f_err'], 'runtime': results_p3p_sampling_hc['runtime'], 'iterations': results_p3p_sampling_hc['info']['iterations'], 'num_inliers': results_p3p_sampling_hc['info']['num_inliers']})

        errors = pd.DataFrame(errors)


        # write to pickle
        with open(results_file, 'wb') as f:
            pickle.dump(Errors_np, f)

        with open(results_file.parent / 'RANSACresults_remove_neighbours_pd.pkl', 'wb') as f:
            pickle.dump(errors, f)

