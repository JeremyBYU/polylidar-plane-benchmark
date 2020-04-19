import logging
import pprint
import pathlib
import time
import csv

import numpy as np
import colorcet as cc

from polylidar_plane_benchmark import (DEFAULT_PPB_FILE, DEFAULT_PPB_FILE_SECONDARY, logger, SYNPEB_ALL_FNAMES, SYNPEB_DIR, SYNPEB_MESHES_DIR,
                                       SYNPEB_DIR_TEST_GT, SYNPEB_DIR_TRAIN_GT, SYNPEB_DIR_TEST_ALL, SYNPEB_DIR_TRAIN_ALL)



# Disable PPB Logger
logger = logging.getLogger("PPB")
logger.setLevel(logging.WARN)

# Enable Train Logger
logger_train = logging.getLogger("PPB_Train")
logger_train.setLevel(logging.WARN)

pp = pprint.PrettyPrinter(indent=4)


def get_fpaths(variance, data='train'):
    if int(variance) == 0:
        base_dir = SYNPEB_DIR_TRAIN_GT if data == "train" else SYNPEB_DIR_TEST_GT
    else:
        base_dir = SYNPEB_DIR_TRAIN_ALL[int(
            variance) - 1] if data == "train" else SYNPEB_DIR_TEST_ALL[int(variance) - 1]

    all_fnames = SYNPEB_ALL_FNAMES
    if int(variance) != 0:
        all_fnames = all_fnames[0:10]

    all_fpaths = [str(base_dir / fname) for fname in all_fnames]
    return all_fpaths



polylidar_kwargs_default = dict(alpha=0.0, lmax=0.1, min_triangles=200,
                                z_thresh=0.1, norm_thresh=0.96, norm_thresh_min=0.96,
                                min_hole_vertices=50, task_threads=4)
level_default = 5


def get_csv_fpath(variance=1, param_index=0):
    this_dir = pathlib.Path(__file__).parent
    data_dir = this_dir.parent.parent / "data"
    results_dir = data_dir / "synpeb_results"

    fname = "synpeb_variance_{}_params_{}.csv".format(variance, param_index)

    fpath = results_dir / fname
    return fpath


def evaluate_with_params(param_set, param_index, variance, counter=None):
    from polylidar_plane_benchmark.utility.helper import (
        convert_planes_to_classified_point_cloud, load_pcd_file,
        extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh)
    from polylidar_plane_benchmark.utility.evaluate import evaluate
    from polylidar_plane_benchmark.utility.helper_mesh import create_meshes_cuda

    from polylidar import Polylidar3D
    from fastga import GaussianAccumulatorS2, IcoCharts

    # from polylidar_plane_benchmark.utility.helper_mesh import lo
    all_fpaths = get_fpaths(variance)

    # print("Here i am ")
    # print(counter)

    # Create Long Lived Objects Only Once
    ga = GaussianAccumulatorS2(level=level_default)  # Fast Gaussian Accumulator
    ico_chart = IcoCharts(level=level_default)  # Used for unwrapping S2 to Image for peak detection
    pl = Polylidar3D(**polylidar_kwargs_default)  # Region Growing and Polygons Extraction

    # logger_train.warn("Working on variance %r, param index %r with %r elements", variance, param_index, len(param_set))
    csv_fpath = get_csv_fpath(variance, param_index)
    with open(csv_fpath, 'w', newline='') as csv_file:
        fieldnames = ['variance', 'fname', 'tcomp',
                      'kernel_size', 'loops_bilateral', 'loops_laplacian', 'sigma_angle',
                      'min_total_weight', 'norm_thresh_min', 'threshold_abs', 'n_gt',
                      'n_ms_all', 'f_weighted_corr_seg', 'f_corr_seg', 'n_corr_seg',
                      'n_over_seg', 'n_under_seg', 'n_missed_seg', 'n_noise_seg',
                      'laplacian', 'bilateral', 'mesh', 'fastga_total', 'fastga_integrate',
                      'fastga_peak', 'polylidar']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        try:
            for fpath in all_fpaths:
                fname = pathlib.Path(fpath).name
                pc_raw, pc_image = load_pcd_file(all_fpaths[0], stride=2)
                logger_train.info("Working on file %s", fpath)
                mesh_kwargs_previous = dict()

                tri_mesh = None
                timings_mesh = None
                for params in param_set:
                    pl.norm_thresh = params['norm_thresh_min']
                    pl.norm_thresh_min = params['norm_thresh_min']
                    t1 = time.perf_counter()
                    mesh_kwargs = {k: params[k]
                                for k in ('loops_laplacian', 'kernel_size', 'loops_bilateral', 'sigma_angle')}
                    # Create Smoothed TriMesh
                    if mesh_kwargs_previous != mesh_kwargs:
                        tri_mesh, mesh_timings = create_meshes_cuda(pc_image, **mesh_kwargs)
                        mesh_kwargs_previous = mesh_kwargs
                    else:
                        logger_train.debug("Reusing previously created mesh!")

                    # Set up kwargs for finding dominiant plane normals (image peak detection)
                    find_peaks_kwargs = dict(threshold_abs=params['threshold_abs'],
                                            min_distance=1, exclude_border=False, indices=False)
                    cluster_kwargs = dict(t=0.10, criterion='distance')
                    average_filter = dict(min_total_weight=params['min_total_weight'])
                    # Extract Dominant Plane Peaks
                    avg_peaks, _, _, _, fastga_timings = extract_all_dominant_plane_normals(tri_mesh, level=level_default,
                                                                                            with_o3d=False, ga_=ga, ico_chart_=ico_chart,
                                                                                            find_peaks_kwargs=find_peaks_kwargs,
                                                                                            cluster_kwargs=cluster_kwargs,
                                                                                            average_filter=average_filter)
                    # Extact Planes and Polygons
                    # MUST NOT BE OPTIMIZED BECAUSE MARL (inside polylidar) CAN"T HANDLE FORKED PROCESSES
                    all_planes, _, _, polylidar_timings = extract_planes_and_polygons_from_mesh(
                        tri_mesh, avg_peaks, filter_polygons=False, pl_=pl, optimized=False)

                    # Aggregate timings
                    all_timings = dict(**mesh_timings, **fastga_timings, **polylidar_timings)

                    # Convert planes to format used for evaluation
                    all_planes_classified = convert_planes_to_classified_point_cloud(all_planes, tri_mesh, avg_peaks)
                    # Evaluate the results. This actually takes the longest amount of time!
                    misc = dict(fname=fname, variance=variance, tcomp=0.80, **params)
                    results_080, auxiliary_080 = evaluate(pc_image, all_planes_classified, tcomp=0.80, misc=misc)
                    results_070, auxiliary_070 = evaluate(pc_image, all_planes_classified, tcomp=0.70, misc=misc)
                    # print(results)
                    logger_train.info("Finished %r", params)

                    full_record_080 = dict(**params, **results_080, **all_timings,
                                        fname=fname, variance=variance, tcomp=0.80)
                    full_record_070 = dict(**params, **results_070, **all_timings,
                                        fname=fname, variance=variance, tcomp=0.70)
                    # Save the record
                    writer.writerow(full_record_080)
                    writer.writerow(full_record_070)

                    t2 = time.perf_counter()
                    elapsed = (t2 - t1) * 1000
                    logger_train.info("One Param Set Took: %.1f", elapsed)
                    # Don't forget to clear the GA
                    ga.clear_count()
                    if counter is not None:
                        with counter.get_lock():
                            counter.value += 1

        except Exception as e:
            logger_train.exception("Something went wrong")
    # logger_train.info("FINISHED: Working on variance %r, param index %r with %r elements", variance, param_index, len(param_set))


def evaluate_with_params_visualize(params):

    from polylidar_plane_benchmark.utility.helper import (
        convert_planes_to_classified_point_cloud, load_pcd_file, paint_planes,
        extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh)
    from polylidar_plane_benchmark.utility.evaluate import evaluate, evaluate_incorrect
    from polylidar_plane_benchmark.utility.helper_mesh import create_meshes_cuda
    from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, create_open_3d_mesh_from_tri_mesh, mark_invalid_planes

    logger.setLevel(logging.INFO)
    # from polylidar_plane_benchmark.utility.helper_mesh import lo
    variance = params['variance']
    all_fpaths = get_fpaths(params['variance'])
    fpaths = [fpath for fpath in all_fpaths if params['fname'] in fpath]
    fpath = fpaths[0]

    # Create Long Lived Objects Only Once
    ga = GaussianAccumulatorS2(level=level_default)  # Fast Gaussian Accumulator
    ico_chart = IcoCharts(level=level_default)  # Used for unwrapping S2 to Image for peak detection
    pl = Polylidar3D(**polylidar_kwargs_default)  # Region Growing and Polygons Extraction

    fname = pathlib.Path(fpath).name
    logger_train.info("Working on file %s", fpath)


    pc_raw, pc_image = load_pcd_file(all_fpaths[0], stride=2)
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    cmap = cc.cm.glasbey_bw
    pcd_raw = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3], cmap=cmap)
    

    pl.norm_thresh = params['norm_thresh_min']
    pl.norm_thresh_min = params['norm_thresh_min']

    t1 = time.perf_counter()
    mesh_kwargs = {k: params[k]
                   for k in ('loops_laplacian', 'kernel_size', 'loops_bilateral', 'sigma_angle')}

    tri_mesh, mesh_timings = create_meshes_cuda(pc_image, **mesh_kwargs)
    mesh_kwargs_previous = mesh_kwargs

    # Set up kwargs for finding dominiant plane normals (image peak detection)
    find_peaks_kwargs = dict(threshold_abs=params['threshold_abs'],
                             min_distance=1, exclude_border=False, indices=False)
    cluster_kwargs = dict(t=0.10, criterion='distance')
    average_filter = dict(min_total_weight=params['min_total_weight'])
    # Extract Dominant Plane Peaks
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, fastga_timings = extract_all_dominant_plane_normals(tri_mesh, level=level_default,
                                                                                                                        with_o3d=True, ga_=ga, ico_chart_=ico_chart,
                                                                                                                        find_peaks_kwargs=find_peaks_kwargs,
                                                                                                                        cluster_kwargs=cluster_kwargs,
                                                                                                                        average_filter=average_filter)
    # Extact Planes and Polygons
    all_planes, all_polygons, all_poly_lines, polylidar_timings = extract_planes_and_polygons_from_mesh(
        tri_mesh, avg_peaks, filter_polygons=False, pl_=pl)

    # Aggregate timings
    all_timings = dict(**mesh_timings, **fastga_timings, **polylidar_timings)

    # Convert planes to format used for evaluation
    all_planes_classified = convert_planes_to_classified_point_cloud(all_planes, tri_mesh, avg_peaks)
    # Evaluate the results. This actually takes the longest amount of time!
    misc = dict(**params)
    # results_080, auxiliary_080 = evaluate(pc_image, all_planes_classified, tcomp=0.80, misc=misc)
    results_070, auxiliary_070 = evaluate_incorrect(pc_image, all_planes_classified, tcomp=params['tcomp'], misc=misc)
    results_070, auxiliary_070 = evaluate(pc_image, all_planes_classified, tcomp=params['tcomp'], misc=misc)
    # print(results)
    logger_train.info("Finished %r", params)

    # create invalid plane markers, green = gt_label_missed, red=ms_labels_noise, blue=gt_label_over_seg, gray=ms_label_under_seg
    invalid_plane_markers = mark_invalid_planes(pc_raw, auxiliary_070, all_planes_classified)
    tri_mesh_o3d = create_open_3d_mesh_from_tri_mesh(tri_mesh)
    tri_mesh_o3d_painted = paint_planes(all_planes_classified, tri_mesh_o3d)

    plot_meshes([pcd_raw, tri_mesh_o3d_painted, *invalid_plane_markers])
