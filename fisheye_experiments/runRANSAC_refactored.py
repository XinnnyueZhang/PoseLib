import poselib
import numpy as np
import pandas as pd
import pycolmap
import pickle
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from ipdb import set_trace

@dataclass
class PoseError:
    """Container for pose estimation errors."""
    rotation_error: float  # in degrees
    translation_error: float  # in mm
    focal_error: float  # in pixels
    runtime: float  # in ms
    iterations: int
    num_inliers: int
    inliers: List[int]


@dataclass
class PoseResult:
    """Container for pose estimation results."""
    quat: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    focal_length: float
    error: PoseError


class PoseErrorCalculator:
    """Handles calculation of pose estimation errors."""
    
    @staticmethod
    def calculate_errors(estimated_pose, ground_truth_pose) -> Tuple[float, float]:
        """Calculate rotation and translation errors between estimated and ground truth poses."""
        # Rotation error in degrees
        rotation_error = np.rad2deg(
            np.arccos(
                np.clip(
                    (np.trace(estimated_pose.R @ ground_truth_pose.rotation.matrix().T) - 1) / 2,
                    -1, 1
                )
            )
        )
        
        # Translation error in mm
        translation_error = np.linalg.norm(estimated_pose.t - ground_truth_pose.translation) * 100
        
        return rotation_error, translation_error


class RANSACConfig:
    """Configuration for RANSAC parameters."""
    
    def __init__(self, threshold: float = 10.0, min_iterations: int = 100, max_iterations: int = 2000):
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
    
    def get_options(self, estimate_focal: bool = True, minimal_solver: str = 'P4Pfr_LM') -> Dict[str, Any]:
        """Get RANSAC options dictionary."""
        return {
            'estimate_focal_length': estimate_focal,
            'estimate_extra_params': False,
            'max_error': self.threshold,
            'minimal_solver': minimal_solver,
            'ransac': {
                'min_iterations': self.min_iterations,
                'max_iterations': self.max_iterations
            },
            'bundle': {
                'verbose': False,
                'loss_scale': self.threshold
            }
        }


class RANSACRunner:
    """Handles RANSAC pose estimation for different solvers."""
    
    def __init__(self, config: RANSACConfig):
        self.config = config
        self.error_calculator = PoseErrorCalculator()
    
    def run_p3p(self, correspondences: Dict[str, Any], query_name: str) -> PoseResult:
        """Run P3P solver with known focal length."""
        corrs = correspondences[query_name]
        
        # Setup fisheye camera and unproject points
        fisheye_cam = poselib.Camera(
            'SIMPLE_FISHEYE', 
            [corrs['cam'].focal_length, 0, 0], 
            corrs['cam'].width, 
            corrs['cam'].height
        )
        points2d_unprojected = fisheye_cam.unproject(corrs['points2D_undistorted'])
        points2d_unprojected = np.array(points2d_unprojected)
        points2d_unprojected = points2d_unprojected[:, :2] / points2d_unprojected[:, 2:3]
        
        # Setup camera for P3P
        camera_dict = {
            'model': 'SIMPLE_PINHOLE',
            'width': corrs['cam'].width,
            'height': corrs['cam'].height,
            'params': [1.0, 0, 0]
        }
        
        # Run P3P estimation
        pose, info = poselib.estimate_absolute_pose(
            points2d_unprojected, 
            corrs['points3D'], 
            camera_dict, 
            self.config.get_options(estimate_focal=False)
        )
        
        return self._create_result(pose, info, corrs, focal_error=0.0)
    
    def run_fisheye_solver(self, correspondences: Dict[str, Any], query_name: str, 
                          solver_name: str) -> PoseResult:
        """Run fisheye solver with focal length estimation."""
        corrs = correspondences[query_name]
        
        # Setup fisheye camera
        camera_dict = {
            'model': 'SIMPLE_FISHEYE',
            'width': corrs['cam'].width,
            'height': corrs['cam'].height,
            'params': [corrs['recalibrated_focal_length'], 0, 0]
        }
        
        # Run fisheye estimation
        pose, info = poselib.estimate_absolute_pose_fisheye(
            corrs['points2D_undistorted'],
            corrs['points3D'],
            camera_dict,
            self.config.get_options(minimal_solver=solver_name)
        )
        
        # Calculate focal length error
        focal_error = abs(pose.camera.focal() - corrs['recalibrated_focal_length'])
        
        return self._create_result(pose, info, corrs, focal_error)
    
    def _create_result(self, pose, info, corrs, focal_error: float) -> PoseResult:
        """Create a PoseResult from pose estimation output."""
        # Calculate pose errors
        rotation_error, translation_error = self.error_calculator.calculate_errors(
            pose.pose, corrs['gtPose']
        )
        
        # Create error object
        error = PoseError(
            rotation_error=rotation_error,
            translation_error=translation_error,
            focal_error=focal_error,
            runtime=info['runtime'] / 1e6,  # Convert to ms
            iterations=info['iterations'],
            num_inliers=info['num_inliers'],
            inliers=info['inliers']
        )
        
        # Create result object
        return PoseResult(
            quat=pose.pose.q,
            translation=pose.pose.t,
            rotation=pose.pose.R,
            focal_length=pose.camera.focal() if hasattr(pose, 'camera') else corrs['cam'].focal_length,
            error=error
        )


class ResultsCollector:
    """Handles collection and storage of RANSAC results."""
    
    def __init__(self):
        self.solvers = [
            'p3p', 'P4Pfr', 'P4Pfr_LM', 'P4Pfr_HC_pose', 'P4Pfr_HC_depth',
            'P3P_sampling_LM', 'P3P_sampling_HC', 'P3P_givenf', 'P5Pfr', 'P5Pfr_LM'
        ]
        # self.solvers = ['P5Pfr_LM', 'P3P_givenf']
    
    def collect_results(self, correspondences: Dict[str, Any], query_list: List[str], 
                       runner: RANSACRunner) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Dict]]:
        """Collect results for all queries and solvers."""
        errors_np = {}
        errors_pd = []
        errors_dict = {}
        
        for query_name in tqdm(query_list):
            query_results = {}
            
            # Run P3P (special case with known focal length)
            p3p_result = runner.run_p3p(correspondences, query_name)
            query_results['p3p'] = p3p_result
            
            # Run all fisheye solvers
            for solver in self.solvers[1:]:  # Skip p3p as it's handled separately
                if solver == 'p3p':
                    continue
                result = runner.run_fisheye_solver(correspondences, query_name, solver)
                query_results[solver] = result

            
            # Store results in different formats
            self._store_query_results(query_name, query_results, errors_np, errors_pd, errors_dict)
        
        return errors_np, pd.DataFrame(errors_pd), errors_dict
    
    def _store_query_results(self, query_name: str, results: Dict[str, PoseResult], 
                           errors_np: Dict, errors_pd: List, errors_dict: Dict):
        """Store results for a single query in all required formats."""
        # Create numpy array for this query
        error_array = []
        for solver in self.solvers:
            if solver in results:
                result = results[solver]
                error_array.extend([
                    result.error.rotation_error,
                    result.error.translation_error,
                    result.error.focal_error,
                    result.error.runtime,
                    result.error.iterations,
                    result.error.num_inliers
                ])
            set_trace()
        errors_np[query_name] = np.array(error_array)
        
        # Store detailed results
        errors_dict[query_name] = results
        
        # Create pandas entries
        for solver, result in results.items():
            errors_pd.append({
                'query': query_name,
                'method': solver,
                'R Error': result.error.rotation_error,
                't Error': result.error.translation_error,
                'f Error': result.error.focal_error,
                'runtime': result.error.runtime,
                'iterations': result.error.iterations,
                'num_inliers': result.error.num_inliers,
                'inliers': result.error.inliers
            })
    
    def save_results(self, results_np: Dict, results_pd: pd.DataFrame, 
                    results_dict: Dict, output_dir: Path, process_name: str):
        """Save results to pickle files."""
        # Save numpy results
        with open(output_dir / f'RANSACresults_{process_name}.pkl', 'wb') as f:
            pickle.dump(results_np, f)
        
        # Save pandas results
        with open(output_dir / f'RANSACresults_pd.pkl', 'wb') as f:
            pickle.dump(results_pd, f)
        
        # Save detailed results
        with open(output_dir / f'RANSACresults_dict.pkl', 'wb') as f:
            pickle.dump(results_dict, f)


class RANSACExperiment:
    """Main class for running RANSAC experiments."""
    
    def __init__(self, process_name: str = "covisible80"):
        self.process_name = process_name
        self.scene_list = [
            "festia_out_corridor", "sportunifront", "parakennus_out", "main_campus",
            "Kitchen_In", "meetingroom", "night_out", "outcorridor", "parakennus", "upstairs"
        ]
        self.gt_dirs = Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD")
    
    def run_experiment(self, threshold: float = 10.0):
        """Run RANSAC experiments for all scenes."""
        config = RANSACConfig(threshold=threshold)
        runner = RANSACRunner(config)
        collector = ResultsCollector()
        
        for scene_name in self.scene_list:
            print(f"Running RANSAC for scene: {scene_name}")
            
            # Setup paths
            data_folder = self.gt_dirs / scene_name / f"processed_{self.process_name}/hloc"
            results_file = data_folder / f'RANSACresults_{self.process_name}.pkl'
            
            # # Skip if results already exist
            # if results_file.exists():
            #     continue
            
            # Load data
            correspondences, query_list = self._load_data(scene_name)
            
            # Run experiments
            errors_np, errors_pd, errors_dict = collector.collect_results(
                correspondences, query_list, runner
            )

            set_trace()
            
            # # Save results
            # collector.save_results(errors_np, errors_pd, errors_dict, data_folder, self.process_name)
    
    def _load_data(self, scene_name: str) -> Tuple[Dict[str, Any], List[str]]:
        """Load correspondences and query list for a scene."""
        data_folder = self.gt_dirs / scene_name / f"processed_{self.process_name}/hloc"
        query_list_dir = self.gt_dirs / scene_name / f"processed_{self.process_name}/data/list_query.txt"
        corrs_dir = self.gt_dirs / scene_name / f"processed_{self.process_name}/Corrs/CorrsDict.pkl"
        
        # Load query list
        with open(query_list_dir, 'r') as f:
            query_list = [line.strip() for line in f.readlines()]
        
        # Load correspondences
        with open(corrs_dir, 'rb') as f:
            correspondences = pickle.load(f)
        
        return correspondences, query_list


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RANSAC experiments for fisheye pose estimation")
    parser.add_argument("--process_name", type=str, default="covisible80", 
                       help="Process name for data organization")
    parser.add_argument("--threshold", type=float, default=10.0, 
                       help="RANSAC threshold")
    args = parser.parse_args()
    
    experiment = RANSACExperiment(args.process_name)
    experiment.run_experiment(args.threshold)


if __name__ == "__main__":
    main()
