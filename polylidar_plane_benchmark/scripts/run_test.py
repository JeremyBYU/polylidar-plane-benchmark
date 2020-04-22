""" This script is used to run tuned hyperparameters on the SynPEB Dataset
Hyperparameters from training were very stable between variance groups. Basically only
Mesh smoothing needs to be changed. 

Note - Polylidar generates *planes* and *polygon*
    Planes are a sets of spatially connected triangles of similar normals
    Polygons are the concave hull and possible interior holes of said planes

SynPEB (and other benchmarks) are **point** based for planes, not polygon based.
For this reason we use the planes returned from polylidar for benchmark evaluation.
This cooresponds to the point indices of the triangles in each plane 

A stride=1 is used for testing, meaning **all** the data (500X500) is used for prediction and evaluation

"""
from polylidar_plane_benchmark.scripts.train_core import evaluate_with_params

# optimal params for stride=1
# params = dict(kernel_size=5, loops_laplacian=2, loops_bilateral=2, sigma_angle=0.1,
#                 min_triangles=1000, norm_thresh_min=0.95, stride=1, predict_loops_laplacian=True)

def main():
    params = dict(kernel_size=3, loops_laplacian=2, loops_bilateral=1, sigma_angle=0.1,
                  min_triangles=250, norm_thresh_min=0.95, stride=2, predict_loops_laplacian=True)
    dataset = 'test'

    evaluate_with_params([params], 0, 1, None, dataset)
    evaluate_with_params([params], 0, 2, None, dataset)
    evaluate_with_params([params], 0, 3, None, dataset)
    evaluate_with_params([params], 0, 4, None, dataset)


if __name__ == "__main__":
    main()

# These parameters are not used, but kept for posterity. **IF** you were to split parameters by variance
# This split would give good results
# var1_params = dict(kernel_size=5, loops_laplacian=2, loops_bilateral=2, sigma_angle=0.1,
#                    min_triangles=1000, norm_thresh_min=0.95, stride=1)
# var2_params = dict(kernel_size=5, loops_laplacian=4, loops_bilateral=2, sigma_angle=0.1,
#                    min_triangles=1000, norm_thresh_min=0.95, stride=1)
# var3_params = dict(kernel_size=5, loops_laplacian=6, loops_bilateral=2, sigma_angle=0.1,
#                    min_triangles=1000, norm_thresh_min=0.95, stride=1)
# var4_params = dict(kernel_size=5, loops_laplacian=8, loops_bilateral=2, sigma_angle=0.1,
#                    min_triangles=1000, norm_thresh_min=0.95, stride=1)