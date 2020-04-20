from polylidar_plane_benchmark.scripts.train_core import evaluate_with_params_visualize

# Error with evaluate_incorrect
# params = {'fname': 'pc_01.pcd', 'variance': 1, 'tcomp': 0.7, 'kernel_size': 3, 'loops_bilateral': 6, 'loops_laplacian': 6, 'sigma_angle': 0.2, 'min_total_weight': 0.01, 'norm_thresh_min': 0.95, 'threshold_abs': 2}

# Last one that broke
# params = {'fname': 'pc_01.pcd', 'tcomp': 0.80, 'variance': 4, 'tcomp': 0.8, 'kernel_size': 5,
#           'loops_bilateral': 8, 'loops_laplacian': 4, 'sigma_angle': 0.2,
#           'norm_thresh_min': 0.95, 'min_triangles': 2000}

# Broken Edge
# File: pc_03.pcd, Variance: 4, Param Index: 0, Params: {'kernel_size': 3, 'loops_bilateral': 0, 'loops_laplacian': 6, 'sigma_angle': 0.1, 'min_triangles': 1000, 'norm_thresh_min': 0.95, 'stride': 1}

    # params = {'fname': 'pc_03.pcd', 'tcomp': 0.80, 'variance': 4, 'tcomp': 0.8, 'kernel_size': 3,
    #           'loops_bilateral': 0, 'loops_laplacian': 6, 'sigma_angle': 0.1,
    #           'norm_thresh_min': 0.95, 'min_triangles': 1000, 'stride':1}
def main():
    params = {'fname': 'pc_02.pcd', 'tcomp': 0.80, 'variance': 4, 'kernel_size': 5,
              'loops_bilateral': 0, 'loops_laplacian': 1, 'sigma_angle': 0.1,
              'norm_thresh_min': 0.95, 'min_triangles': 1000, 'stride':1}
    evaluate_with_params_visualize(params)


if __name__ == "__main__":
    main()
