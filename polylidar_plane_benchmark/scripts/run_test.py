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

# semi-optimal params for stride=2
# params = dict(kernel_size=3, loops_laplacian=2, loops_bilateral=1, sigma_angle=0.1,
#               min_triangles=250, norm_thresh_min=0.95, stride=2, predict_loops_laplacian=True)

def main():
    params = dict(kernel_size=5, loops_laplacian=2, loops_bilateral=2, sigma_angle=0.1,
                    min_triangles=1000, norm_thresh_min=0.95, stride=1, predict_loops_laplacian=True)
    dataset = 'test'
    print("Evaluating Variance 1")
    evaluate_with_params([params], 0, 1, None, dataset)
    print("Evaluating Variance 2")
    evaluate_with_params([params], 0, 2, None, dataset)
    print("Evaluating Variance 3")
    evaluate_with_params([params], 0, 3, None, dataset)
    print("Evaluating Variance 4")
    evaluate_with_params([params], 0, 4, None, dataset)


if __name__ == "__main__":
    main()


# Stride 1 
# [4 rows x 16 columns]
# n_gt                   42.491667
# n_ms_all               25.316667
# f_weighted_corr_seg     0.762061
# rmse                    0.009151
# f_corr_seg              0.470380
# n_corr_seg             18.308333
# n_over_seg              0.266667
# n_under_seg             0.400000
# n_missed_seg           23.033333
# n_noise_seg             6.058333
# laplacian               1.227258
# bilateral               2.994489
# mesh                    8.497396
# fastga_total            6.612652
# polylidar              14.829927
# dtype: float64


# Stride 2
# n_gt                   42.491667
# n_ms_all               26.400000
# f_weighted_corr_seg     0.697639
# rmse                    0.009437
# f_corr_seg              0.392567
# n_corr_seg             14.658333
# n_over_seg              0.950000
# n_under_seg             0.341667
# n_missed_seg           26.150000
# n_noise_seg             8.908333
# laplacian               0.472313
# bilateral               0.715748
# mesh                    1.793931
# fastga_total            2.583590
# polylidar               4.291157
# dtype: float64

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
