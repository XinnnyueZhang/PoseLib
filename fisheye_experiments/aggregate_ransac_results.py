import argparse
from pathlib import Path
import pickle
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_scene_results(scene_dir: Path, process_name: str, threshold: float | None) -> Dict[str, np.ndarray]:
    """Load the per-query error arrays for one scene.

    Tries both naming schemes seen in the codebase:
    - RANSACresults_{threshold}.pkl (runRANSAC.py)
    - RANSACresults_{process_name}.pkl (runRANSAC_refactored.py)
    """
    hloc_dir = scene_dir / f"processed_{process_name}" / "hloc"
    candidates = []
    if threshold is not None:
        candidates.append(hloc_dir / f"RANSACresults_{threshold}.pkl")
    candidates.append(hloc_dir / f"RANSACresults_{process_name}.pkl")

    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No results file found in {hloc_dir} among candidates: {[str(p) for p in candidates]}")


def collect_overall_results(
    gt_root: Path,
    scenes: List[str],
    process_name: str,
    threshold: float | None,
    method_names: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Concatenate per-scene results into a single matrix and return also per-scene matrices.

    Returns:
    - results_overall: (num_queries_all, num_methods*6)
    - results_all: dict scene -> array
    """
    results_all: Dict[str, np.ndarray] = {}
    results_overall_list: List[np.ndarray] = []

    for scene in scenes:
        errors_map = load_scene_results(gt_root / scene, process_name, threshold)
        # Convert dict of query_name -> flat vector to array
        results_scene = np.stack([errors_map[q] for q in errors_map.keys()])
        # Validate shape: each method contributes 6 metrics
        expected_len = len(method_names) * 6
        if results_scene.shape[1] != expected_len:
            raise ValueError(
                f"Scene {scene}: expected {expected_len} metrics per query (methods*6), got {results_scene.shape[1]}."
            )
        results_all[scene] = results_scene
        results_overall_list.append(results_scene)

    results_overall = np.concatenate(results_overall_list, axis=0)
    return results_overall, results_all


def compute_summary_tables(results_overall: np.ndarray, method_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Compute median/mean tables and AUC tables.

    Metric order per method (6 columns per method):
    [R Error (deg), t Error (cm), f Error (px), Runtime (ms), Iterations, Num Inliers]
    """
    def show_errors_row(vector: np.ndarray) -> pd.DataFrame:
        headers = [
            "R Error (deg)",
            "t Error (cm)",
            "f Error (px)",
            "Runtime (ms)",
            "Iterations",
            "Num Inliers",
        ]
        table = np.reshape(vector, (len(method_names), int(vector.shape[0] / len(method_names))))
        return pd.DataFrame(table, columns=headers, index=method_names)

    median_table = show_errors_row(np.median(results_overall, axis=0))
    mean_table = show_errors_row(np.mean(results_overall, axis=0))

    # AUC for (R, t)
    R_thr = [1, 1, 2, 5, 5]
    t_thr = [1, 2, 5, 5, 10]
    # AUC for (t, f)
    f_thr = [1, 2, 5, 10, 50]

    def get_auc(errors: np.ndarray, idx_a: int, idx_b: int, thrs_a: List[float], thrs_b: List[float]) -> List[float]:
        assert len(thrs_a) == len(thrs_b)
        n = errors.shape[0]
        auc_vals: List[float] = []
        for j in range(len(thrs_a)):
            count = 0
            for i in range(n):
                if errors[i, idx_a] < thrs_a[j] and errors[i, idx_b] < thrs_b[j]:
                    count += 1
            auc_vals.append(count / n)
        return auc_vals

    auc_rt_rows = []
    auc_tf_rows = []
    mean_time = np.mean(results_overall, axis=0)
    for m_idx in range(len(method_names)):
        base = m_idx * 6
        auc_rt = get_auc(results_overall, base + 0, base + 1, R_thr, t_thr)
        auc_tf = get_auc(results_overall, base + 1, base + 2, t_thr, f_thr)
        auc_rt_rows.append(
            {
                "method": method_names[m_idx],
                f"{R_thr[0]}deg, {t_thr[0]}cm": auc_rt[0] * 100,
                f"{R_thr[1]}deg, {t_thr[1]}cm": auc_rt[1] * 100,
                f"{R_thr[2]}deg, {t_thr[2]}cm": auc_rt[2] * 100,
                f"{R_thr[3]}deg, {t_thr[3]}cm": auc_rt[3] * 100,
                f"{R_thr[4]}deg, {t_thr[4]}cm": auc_rt[4] * 100,
                "mean time (ms)": mean_time[base + 3],
            }
        )
        auc_tf_rows.append(
            {
                "method": method_names[m_idx],
                f"{t_thr[0]}cm, {f_thr[0]}px": auc_tf[0] * 100,
                f"{t_thr[1]}cm, {f_thr[1]}px": auc_tf[1] * 100,
                f"{t_thr[2]}cm, {f_thr[2]}px": auc_tf[2] * 100,
                f"{t_thr[3]}cm, {f_thr[3]}px": auc_tf[3] * 100,
                f"{t_thr[4]}cm, {f_thr[4]}px": auc_tf[4] * 100,
                "mean time (ms)": mean_time[base + 3],
            }
        )

    auc_rt_df = pd.DataFrame(auc_rt_rows).set_index("method")
    auc_tf_df = pd.DataFrame(auc_tf_rows).set_index("method")

    return {
        "median": median_table,
        "mean": mean_table,
        "auc_rt": auc_rt_df,
        "auc_tf": auc_tf_df,
    }


def plot_overall_figures(results_overall: np.ndarray, method_names: List[str], out_dir: Path) -> None:
    """Generate CDFs and boxplots, saving to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    error_names = [
        "R Error (deg)",
        "t Error (cm)",
        "f Error (px)",
        "Runtime (ms)",
        "Iterations",
        "Num Inliers",
    ]
    error_limits = [1, 10, 10, 200, 300, 2000]

    # CDFs
    plt.figure(figsize=(12, 7))
    for i in range(len(error_names)):
        plt.subplot(2, 3, i + 1)
        for j, name in enumerate(method_names):
            sns.ecdfplot(results_overall[:, i + j * 6], label=name if i == 0 else None)
        plt.xlabel(error_names[i])
        plt.xlim(0, error_limits[i])
        plt.ylabel("CDF")
    plt.legend(method_names, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "cdf_all_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Time boxplot
    time_idx = [i * 6 + 3 for i in range(len(method_names))]
    time_overall = results_overall[:, time_idx]
    plt.figure(figsize=(12, 6))
    plt.boxplot(time_overall, labels=method_names)
    plt.xlabel("Methods")
    plt.ylabel("Time (ms)")
    plt.title("Runtime Boxplot")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_runtime.png", dpi=200)
    plt.close()

    # Iterations boxplot
    iter_idx = [i * 6 + 4 for i in range(len(method_names))]
    iter_overall = results_overall[:, iter_idx]
    plt.figure(figsize=(12, 6))
    plt.boxplot(iter_overall, labels=method_names)
    plt.xlabel("Methods")
    plt.ylabel("Iterations")
    plt.title("Iterations Boxplot")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_iterations.png", dpi=200)
    plt.close()


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSVs
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv")


def main():
    parser = argparse.ArgumentParser(description="Aggregate RANSAC results over all scenes and visualize errors")
    parser.add_argument("--gt_root", type=Path, default=Path("/home2/xi5511zh/Xinyue/Datasets/Fisheye_FIORD"))
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=[
            "festia_out_corridor",
            "sportunifront",
            "parakennus_out",
            "main_campus",
            "Kitchen_In",
            "meetingroom",
            "night_out",
            "outcorridor",
            "parakennus",
            "upstairs",
        ],
    )
    parser.add_argument("--process_name", type=str, default="covisible80")
    parser.add_argument("--threshold", type=float, default=None, help="If provided, prefer RANSACresults_{threshold}.pkl")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home2/xi5511zh/Xinyue/Projects/PoseLib/fisheye_experiments/aggregated_outputs"),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "recalibrator",
            "P4Pfr",
            "P4Pfr_LM",
            "P5Pfr",
            "P5Pfr_LM",
            "P4Pfr_HC_pose",
            "P4Pfr_HC_depth",
            "p3p",
            "P3P_sampling_LM",
            "P3P_sampling_HC",
            "p3p_given_gtf",
            "p3p_given_anyCalibf1",
            "p3p_given_anyCalibf4",
        ],
        help="Method names order matching per-query vectors (6 metrics per method)",
    )
    args = parser.parse_args()

    results_overall, _ = collect_overall_results(args.gt_root, args.scenes, args.process_name, args.threshold, args.methods)
    tables = compute_summary_tables(results_overall, args.methods)
    save_tables(tables, args.out_dir)
    plot_overall_figures(results_overall, args.methods, args.out_dir)

    # Also save the raw combined matrix for future use
    np.save(args.out_dir / "results_overall.npy", results_overall)


if __name__ == "__main__":
    main()


