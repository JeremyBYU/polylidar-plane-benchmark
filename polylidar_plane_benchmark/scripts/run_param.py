""" This script is used to run just **one** parameter configuration for debugging purposes

"""
from polylidar_plane_benchmark.scripts.train_core import evaluate_with_params_visualize

def main():
    params = {'fname': 'pc_02.pcd', 'tcomp': 0.80, 'variance': 4, 'kernel_size': 5,
              'loops_bilateral': 0, 'loops_laplacian': 1, 'sigma_angle': 0.1,
              'norm_thresh_min': 0.95, 'min_triangles': 1000, 'stride':1}
    evaluate_with_params_visualize(params)


if __name__ == "__main__":
    main()
