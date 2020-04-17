from polylidar_plane_benchmark.scripts.train_core import evaluate_with_params_visualize

def main():
    params = {'fname': 'pc_01.pcd', 'variance': 2, 'tcomp': 0.7, 'kernel_size': 3, 'loops_bilateral': 6, 'loops_laplacian': 6, 'sigma_angle': 0.2, 'min_total_weight': 0.01, 'norm_thresh_min': 0.95, 'threshold_abs': 2}
    evaluate_with_params_visualize(params)

if __name__ == "__main__":
    main()