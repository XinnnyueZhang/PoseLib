import argparse
from pathlib import Path
import pickle
import numpy as np

from hloc import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from hloc.pipelines.Cambridge.utils import evaluate

import pycolmap
import poselib

from ipdb import set_trace


        
def getQueryList(test_list_dir, query_list_dir, query_info_dir, gt_sfm_dir, ref_sfm_dir, num_remove_neighbours=10):
    # keep 2n frames -> remove n frames -> keep 1 as query image -> remove n-1 frames -> keep 2n frames -> ...

    if test_list_dir.exists() and query_list_dir.exists() and query_info_dir.exists() and (ref_sfm_dir / "images.bin").exists():
        return

    gt_sfm = pycolmap.Reconstruction(gt_sfm_dir)

    # sort the image by names
    image_list = list(gt_sfm.images.keys())
    image_list.sort()

    query_list = []
    remove_list = []
    keep_list = []


    idx = 0
    remove_count = 0
    keep_count = 0
    while idx < len(image_list):
        if keep_count < 2 * num_remove_neighbours:
            keep_list.append(image_list[idx])
            keep_count += 1
        else:
            if remove_count < 2 * num_remove_neighbours:
                if remove_count == num_remove_neighbours:
                    query_list.append(image_list[idx])

                remove_list.append(image_list[idx])
                remove_count += 1
            else:
                remove_count = 0
                keep_count = 0
                keep_list.append(image_list[idx])

        idx += 1

    test_list = []
    querys = []
    query_info = {}
    for imageID in query_list:
        image = gt_sfm.images[imageID]
        camera = gt_sfm.cameras[image.camera_id]

        # save query list without intrinsics
        test_list.append(image.name)
        
        # save query_list with intrinsics
        p = [image.name, camera.model.name, camera.width, camera.height] + camera.params.tolist()
        querys.append(" ".join(map(str, p)))

        # save gt pose and imageID
        query_info[image.name] = {
            'Pose': gt_sfm.images[imageID].cam_from_world,
            'imageID': imageID,
        }


    with open(test_list_dir, "w") as f:
        f.write("\n".join(test_list))

    with open(query_list_dir, "w") as f:
        f.write("\n".join(querys))

    with open(query_info_dir, "wb") as f:
        pickle.dump(query_info, f)

    # save new sfm without query images
    for imageID in remove_list:
        gt_sfm.deregister_image(imageID)
    gt_sfm.write(ref_sfm_dir)
    


def run_scene(gt_dir, results, num_covis, num_loc, num_remove_neighbours=5):
    
    images = gt_dir / "images"
    gt_sfm = gt_dir / "colmap/model"

    ref_sfm = gt_dir / "processed_remove_neighbours/colmap" # to be filled
    query_dir = gt_dir / "processed_remove_neighbours/data"
    test_list = query_dir / "list_query.txt"
    corrs_dir = gt_dir / "processed_remove_neighbours/Corrs"
    outputs_dir = gt_dir / "processed_remove_neighbours/hloc"

    results_log_dir = outputs_dir / "results.txt_logs.pkl"
    ref_sfm_retriangulated = outputs_dir / "sfm_superpoint+superglue"
    query_list = outputs_dir / "query_list_with_intrinsics.txt"
    sfm_pairs = outputs_dir / f"pairs-db-covis{num_covis}.txt"
    loc_pairs = outputs_dir / f"pairs-query-netvlad{num_loc}.txt"

    ref_sfm.mkdir(exist_ok=True, parents=True)
    query_dir.mkdir(exist_ok=True, parents=True)
    corrs_dir.mkdir(exist_ok=True, parents=True)
    outputs_dir.mkdir(exist_ok=True, parents=True)

    getQueryList(test_list, query_list, query_dir / "queryDict.pkl", gt_sfm, ref_sfm, num_remove_neighbours)

    feature_conf = {
        "output": "feats-superpoint",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    matcher_conf = match_features.confs["superglue"]
    retrieval_conf = extract_features.confs["netvlad"]


    with open(test_list, "r") as f:
        query_seqs = {q.split("/")[0] for q in f.read().rstrip().split("\n")}

    global_descriptors = extract_features.main(retrieval_conf, images, outputs_dir)

    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        db_model=ref_sfm,
        query_prefix=query_seqs,
    )

    features = extract_features.main(feature_conf, images, outputs_dir, as_half=True)

    pairs_from_covisibility.main(ref_sfm, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs_dir
    )

    # Triangulate 3D points from known camera poses
    triangulation.main(
        ref_sfm_retriangulated, ref_sfm, images, sfm_pairs, features, sfm_matches
    )

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs_dir
    )

    localize_sfm.main(
        ref_sfm_retriangulated,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=True,
    )

    get_correspondences_all(ref_sfm_retriangulated, results_log_dir, corrs_dir, query_dir)


def undistort_points_recalibrator(points2D, cam):

    points2D_unprojected = np.array(cam.cam_from_img(points2D))

    points2D_unprojected = np.concatenate((points2D_unprojected, np.ones((points2D.shape[0], 1))), axis=1)

    openCVFisheyeCamera = poselib.Camera('OPENCV_FISHEYE', [cam.focal_length_x, cam.focal_length_y, 0, 0, 0, 0, 0, 0], cam.width, cam.height)

    points2D_projected_unDistorted = openCVFisheyeCamera.project(points2D_unprojected)

    cam_source = {'model': 'OPENCV_FISHEYE', 'params': [cam.focal_length_x, cam.focal_length_y, 0, 0, 0, 0, 0, 0], 'width': cam.width, 'height': cam.height}
    cam_target = {'model': 'SIMPLE_FISHEYE', 'params': [cam.focal_length, 0, 0], 'width': cam.width, 'height': cam.height}

    recalibrated_cam, stats = poselib.recalibrate_camera(points2D_projected_unDistorted, cam_source, cam_target)

    points2D_simplefisheye = recalibrated_cam.project(points2D_unprojected)

    return np.array(points2D_simplefisheye), recalibrated_cam, points2D_unprojected

# get the correspondences
def get_correspondences_all(ref_sfm_dir, log_dir, out_dir, query_dir):

    # if (out_dir / "CorrsDict.pkl").exists():
    #     return

    # load sfm model
    reference_sfm = pycolmap.Reconstruction(ref_sfm_dir)
    
    # load logs including 2D pts and 3D pts index
    with open(log_dir, 'rb') as f:
        logs = pickle.load(f)

    # get the query images list
    query_imgs_list = list(logs['loc'].keys())

    # load gt pose
    with open(query_dir / "queryDict.pkl", 'rb') as f:
        gtPose = pickle.load(f)
        
    inliers_list = []

    CorrsDict = {}

    for query_name in query_imgs_list:

        inliers_array = []
        inliers_array.append(query_name)

        # get the 2D points
        points2D = logs['loc'][query_name]['keypoints_query']

        # get the 3D points
        points3D_id = logs['loc'][query_name]['points3D_ids']
        
        points3D = [reference_sfm.points3D[j].xyz for j in points3D_id]

        points3D = np.vstack(points3D)

        nInliers = logs['loc'][query_name]['PnP_ret']['num_inliers']
        inliers_array.append(nInliers)
        inliers_array.append(nInliers/points3D.shape[0])

        PnP_ret = logs['loc'][query_name]['PnP_ret']['cam_from_world']

        gtPose_img = gtPose[query_name]['Pose']
        Rgt = gtPose_img.rotation.matrix()

        # cam_trans = - np.matmul(cam_rot, cam_trans)
        tgt = gtPose_img.translation

        inliers_list.append(inliers_array)

        CorrsDict[query_name] = {}
        CorrsDict[query_name]['points2D'] = points2D
        CorrsDict[query_name]['points3D'] = points3D
        CorrsDict[query_name]['PnP_ret'] = PnP_ret
        CorrsDict[query_name]['gtPose'] = gtPose_img
        CorrsDict[query_name]['Rgt'] = Rgt
        CorrsDict[query_name]['tgt'] = tgt
        CorrsDict[query_name]['nInliers'] = nInliers
        CorrsDict[query_name]['imageID'] = gtPose[query_name]['imageID']

        if "cam1" in query_name:
            CorrsDict[query_name]['cam'] = reference_sfm.cameras[1]
        elif "cam2" in query_name:
            CorrsDict[query_name]['cam'] = reference_sfm.cameras[2]
        else:
            # error
            raise ValueError(f"Camera name not found in {query_name}")
        
        points2D_undistorted, recalibrated_cam, points2D_unprojected = undistort_points_recalibrator(points2D, CorrsDict[query_name]['cam'])
        CorrsDict[query_name]['points2D_undistorted'] = points2D_undistorted
        CorrsDict[query_name]['recalibrated_focal_length'] = recalibrated_cam.focal()

    with open(Path(out_dir) / f"CorrsDict.pkl", 'wb') as f:
        pickle.dump(CorrsDict, f)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )

    parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )

    parser.add_argument(
        "--num_remove_neighbours",
        type=int,
        default=5,
        help="Number of neighbours to remove, default: %(default)s",
    )

    args = parser.parse_args()


    gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")

    # "festia_out_corridor", 
    SCENES = ["sportunifront", "parakennus_out", "main_campus",
              "Kitchen_In", "meetingroom", "night_out", "outcorridor", "parakennus", "upstairs"]

    # scene = SCENES[0]
    for scene in SCENES:
        results = gt_dirs / scene / "processed_remove_neighbours/hloc/results.txt"
        results.parent.mkdir(exist_ok=True, parents=True)

        if args.overwrite or not results.exists():
        # if True:
            run_scene(
                gt_dirs / scene,
                results,
                args.num_covis,
                args.num_loc,
                args.num_remove_neighbours,
            )
            

        logger.info(f'Evaluate scene "{scene}".')
        evaluate(
            gt_dirs / scene / "colmap/model",
            results,
            gt_dirs / scene / "processed_remove_neighbours/data/list_query.txt",
            ext=".bin",
    )
