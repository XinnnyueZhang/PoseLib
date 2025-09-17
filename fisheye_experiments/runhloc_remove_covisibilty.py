import argparse
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from hloc import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation
)
from hloc.pipelines.Cambridge.utils import evaluate
from hloc.utils.read_write_model import read_model

import pycolmap
import poselib

from ipdb import set_trace


        
def getQueryList(test_list_dir, query_list_dir, query_info_dir, gt_sfm_dir, ref_sfm_dir, nSample = 50, remove_covisibilty=80, camera=1):

    if test_list_dir.exists() and query_list_dir.exists() and query_info_dir.exists() and (ref_sfm_dir / "images.bin").exists():
        return

    gt_sfm = pycolmap.Reconstruction(gt_sfm_dir)

    query_list = []
    remove_list = []
    keep_list = []

    # keep images of camera only
    image_list_all = list(gt_sfm.images.keys())
    image_list = image_list_all.copy()
    for imageID in image_list_all:
        image = gt_sfm.images[imageID]
        if image.camera_id != camera:
            image_list.remove(imageID)
            remove_list.append(imageID)
            
    # random select nSample images
    query_list = list(np.random.choice(image_list, nSample, replace=False))

    # per image remove images with covisibility less than remove_covisibilty
    cameras_, images_, points3D_ = read_model(gt_sfm_dir)
    for imageID in tqdm(query_list):
        matched = images_[imageID].point3D_ids != -1
        points3D_covis = images_[imageID].point3D_ids[matched]

        covis = defaultdict(int)
        for point_id in points3D_covis:
            for image_covis_id in points3D_[point_id].image_ids:
                if image_covis_id != imageID:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            continue

        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])

        for id in covis_ids:
            # if covis[id] > remove_covisibilty*sum(matched)/100:
            if covis[id] > min(2000, remove_covisibilty*sum(matched)/100):
                if id not in remove_list:
                    remove_list.append(id)
                    
    keep_list = list(set(image_list) - set(remove_list) - set(query_list))

    # process for hloc
    test_list = []
    querys = []
    query_info = {}
    for imageID in query_list:
        image = gt_sfm.images[imageID]
        camera_ = gt_sfm.cameras[image.camera_id]

        # save query list without intrinsics
        test_list.append(image.name)
        
        # save query_list with intrinsics
        p = [image.name, camera_.model.name, camera_.width, camera_.height] + camera_.params.tolist()
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
    


def run_scene(gt_dir, results, num_covis, num_loc, nSample=50, remove_covisibilty=80, camera=1):
    
    images = gt_dir / "images"
    gt_sfm = gt_dir / "colmap/model"

    ref_sfm = gt_dir / f"processed_covisible{remove_covisibilty}/colmap" # to be filled
    query_dir = gt_dir / f"processed_covisible{remove_covisibilty}/data"
    test_list = query_dir / "list_query.txt"
    corrs_dir = gt_dir / f"processed_covisible{remove_covisibilty}/Corrs"
    outputs_dir = gt_dir / f"processed_covisible{remove_covisibilty}/hloc"

    results_log_dir = outputs_dir / "results.txt_logs.pkl"
    ref_sfm_retriangulated = outputs_dir / "sfm_superpoint+superglue"
    query_list = outputs_dir / "query_list_with_intrinsics.txt"
    sfm_pairs = outputs_dir / f"pairs-db-covis{num_covis}.txt"
    loc_pairs = outputs_dir / f"pairs-query-netvlad{num_loc}.txt"

    ref_sfm.mkdir(exist_ok=True, parents=True)
    query_dir.mkdir(exist_ok=True, parents=True)
    corrs_dir.mkdir(exist_ok=True, parents=True)
    outputs_dir.mkdir(exist_ok=True, parents=True)

    getQueryList(test_list, query_list, query_dir / "queryDict.pkl", gt_sfm, ref_sfm, nSample, remove_covisibilty, camera)

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

    # use previous superpoint features
    feature_dir = Path(f'/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD/{scene}/processed_covisible80/hloc/feats-superpoint.h5')

    features = extract_features.main(feature_conf, images, outputs_dir, as_half=True, feature_path=feature_dir)


    pairs_from_covisibility.main(ref_sfm, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, features, outputs_dir
    )

    # Triangulate 3D points from known camera poses
    triangulation.main(
        ref_sfm_retriangulated, ref_sfm, images, sfm_pairs, features, sfm_matches
    )

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, features, outputs_dir
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
        "--num_remove_covisibilty",
        type=int,
        default=80,
        help="proportion of covisibilty to remove, default: %(default)s",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="which camera to use, default: %(default)s",
    )

    parser.add_argument(
        "--nSample",
        type=int,
        default=50,
        help="Number of images to sample, default: %(default)s",
    )

    parser.add_argument(
        "--sceneID",
        type=int,
        default=0,
        help="List of images to sample, default: %(default)s",
    )

    args = parser.parse_args()

    # set seed
    np.random.seed(42)

    gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")

    SCENES = ["festia_out_corridor", "sportunifront", "parakennus_out", "main_campus",
              "Kitchen_In", "meetingroom", "night_out", "outcorridor", "parakennus", "upstairs"]

    # sample 30 frames from [2,4,5,6,7,8]
    # sample 50 frames from [0,1,3]

    scene = SCENES[args.sceneID]
    # for scene in SCENES:
    results = gt_dirs / scene / f"processed_covisible{args.num_remove_covisibilty}/hloc/results.txt"
    results.parent.mkdir(exist_ok=True, parents=True)

    if args.overwrite or not results.exists():
    # if True:
        run_scene(
            gt_dirs / scene,
            results,
            args.num_covis,
            args.num_loc,
            args.nSample,
            args.num_remove_covisibilty,
            args.camera,
        )
        

    logger.info(f'Evaluate scene "{scene}".')
    evaluate(
        gt_dirs / scene / "colmap/model",
        results,
        gt_dirs / scene / f"processed_covisible{args.num_remove_covisibilty}/data/list_query.txt",
        ext=".bin",
    )
