import random
import time
import warnings
import math
import gc
import pathlib
warnings.filterwarnings("ignore", message="Optimal rotation is not uniquely or poorly defined ")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pypcd.pypcd as pypcd
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from polylidar import extract_point_cloud_from_float_depth, extract_tri_mesh_from_organized_point_cloud, MatrixDouble, Polylidar3D
from polylidar.polylidarutil.open3d_util import construct_grid, create_lines, flatten
from polylidar.polylidarutil.plane_filtering import filter_planes_and_holes

from polylidar_plane_benchmark.utility.o3d_util import (
    create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, open_3d_mesh_to_tri_mesh, assign_vertex_colors, get_colors, assign_some_vertex_colors)
from polylidar_plane_benchmark.utility.helper_mesh import create_meshes_cuda_with_o3d
from polylidar_plane_benchmark import logger

from fastga import GaussianAccumulatorS2, MatX3d, IcoCharts
from fastga.peak_and_cluster import find_peaks_from_ico_charts
from fastga.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals

import organizedpointfilters as opf
import organizedpointfilters.cuda as opf_cuda
from organizedpointfilters import Matrix3f, Matrix3fRef
import colorcet as cc

from polylidar_plane_benchmark.utility.geometry import convert_to_shapely_geometry_in_image_space, rasterize_polygon, extract_image_coordinates

# Set the random seeds for determinism
random.seed(0)
np.random.seed(0)

splits_stride_1 = np.array([.0002, .0004, .0006])
loops_laplacian_stride_1 = [2, 4, 6, 8]
splits_stride_2 = np.array([.0002, .0003, .00045])
loops_laplacian_stride_2 = [2, 4, 6, 8]


def predict_loops_laplacian(pc_image, stride=1, samples=100, sample_size=2):
    """Very simple model that estimates point cloud noise and predicts the number of laplacian smoothing iterations needed

    Arguments:
        pc_image {ndarray} -- Organized Point Cloud

    Keyword Arguments:
        stride {int} -- How much did you downsample the image (default: {1})
        samples {int} -- How many samples to estimate the point cloud noise (default: {100})
        sample_size {int} -- How big should each sample by (2X2 section) (default: {2})
    """
    assert stride < 3, "Can only handle stride == 1|2;"
    if stride == 1:
        splits = splits_stride_1
        loops = loops_laplacian_stride_1
    else:
        splits = splits_stride_2
        loops = loops_laplacian_stride_2
    noise = estimate_pc_noise(pc_image, samples=samples, sample_size=sample_size)
    idx = np.searchsorted(splits, noise)
    predicted_loops = loops[idx]
    return predicted_loops


def estimate_pc_noise(pc_image, samples=100, sample_size=2):
    """Estimates point cloud noise by sampling
    Looks at a sample_size X sample_size window and computes the cross covariance
    100 samples are taken uniformly from the organized point cloud
    Note that this would be so much faster using C++, just didn't have time to do it
    It currently takes about 1-1.5 ms on my machine.

    Arguments:
        pc_image {[type]} -- [description]

    Keyword Arguments:
        samples {int} -- [description] (default: {100})
        sample_size {int} -- [description] (default: {2})

    Returns:
        [float] -- noise
    """
    rows = pc_image.shape[0] - sample_size * 2
    cols = pc_image.shape[1] - sample_size * 2

    row_skip = int(rows / math.sqrt(samples))
    col_skip = int(cols / math.sqrt(samples))

    pc_size = sample_size * sample_size

    t1 = time.perf_counter()
    matrix_list = []
    for row in range(sample_size, rows, row_skip):
        for col in range(sample_size, cols, col_skip):
            pc_sampled = np.ascontiguousarray(
                pc_image[row:row + sample_size, col:col + sample_size, :3].reshape((pc_size, 3)).T)
            matrix_list.append(pc_sampled)

    t2 = time.perf_counter()

    elapsed_time = (t2 - t1) * 1000
    logger.debug("Noise Estimation - Gathering data took (ms): %.3f ", elapsed_time)

    noise = np.zeros(shape=(len(matrix_list),))
    for i, pc_matrix in enumerate(matrix_list):
        # t3 = time.perf_counter_ns()
        pc_matrix_normalized = pc_matrix - np.mean(pc_matrix, axis=1)[:, None]
        # t4 = time.perf_counter_ns()
        pseudo_noise = pc_matrix_normalized @ pc_matrix_normalized.T
        # t5 = time.perf_counter_ns()
        noise[i] = np.sum(pseudo_noise)
        # t6= time.perf_counter_ns()
        # logger.info("normalized : %1f, Matrix mult: %.1f, sum: %.1f ", (t4 - t3), (t5-t4), (t6-t5))
    t3 = time.perf_counter()
    median_noise = np.median(noise)
    elapsed_time = (t3 - t2) * 1000
    logger.debug("Noise Estimation - Cross Covariance Took (ms): %.3f ", elapsed_time)

    return median_noise


def load_pcd_and_meshes(input_file, stride=2, loops=5, _lambda=0.5, loops_bilateral=0, kernel_size=3, **kwargs):
    """Load PCD and Meshes
    """
    pc_raw, pc_image = load_pcd_file(input_file, stride)

    # Get just the points, no intensity
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    cmap = cc.cm.glasbey_bw
    pcd_raw = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3], cmap=cmap)
    # tri_mesh, tri_mesh_o3d = create_meshes(pc_points, stride=stride, loops=loops)
    tri_mesh, tri_mesh_o3d, timings = create_meshes_cuda_with_o3d(pc_image, loops_laplacian=loops, _lambda=_lambda,
                                                                  loops_bilateral=loops_bilateral, kernel_size=kernel_size, sigma_angle=0.1, **kwargs)

    logger.info("Visualizing Point Cloud - Size: %dX%d ; # Points: %d; # Triangles: %d",
                pc_image.shape[0], pc_image.shape[1], pc_raw.shape[0], np.asarray(tri_mesh.triangles).shape[0])

    return pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, timings


def filter_and_create_open3d_polygons(points, polygons, rm=None, line_radius=0.005,
                                      config_pp=dict(filter=dict(hole_area=dict(min=0.025, max=100.0), hole_vertices=dict(min=6), plane_area=dict(min=0.05)),
                                                     positive_buffer=0.00, negative_buffer=0.00, simplify=0.0), **kwargs):
    " Apply polygon filtering algorithm, return Open3D Mesh Lines "
    # config_pp = dict(filter=dict(hole_area=dict(min=0.00, max=100.0), hole_vertices=dict(min=6), plane_area=dict(min=0.0001)),
    #                  positive_buffer=0.00, negative_buffer=0.0, simplify=0.00)
    t1 = time.perf_counter()
    planes, obstacles = filter_planes_and_holes(polygons, points, config_pp, rm=rm)
    t2 = time.perf_counter()
    logger.info("Plane Filtering Took (ms): %.2f", (t2 - t1) * 1000)
    all_poly_lines = create_lines(planes, obstacles, line_radius=line_radius)
    return all_poly_lines, (t2 - t1) * 1000


def extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks,
                                          polylidar_kwargs=dict(alpha=0.0, lmax=0.1, min_triangles=2000,
                                                                z_thresh=0.1, norm_thresh=0.95, norm_thresh_min=0.95, min_hole_vertices=50, task_threads=4),
                                          filter_polygons=True, pl_=None, optimized=True, **kwargs):

    if pl_ is not None:
        pl = pl_
    else:
        pl = Polylidar3D(**polylidar_kwargs)

    avg_peaks_mat = MatrixDouble(avg_peaks)
    t0 = time.perf_counter()
    if optimized:
        all_planes, all_polygons = pl.extract_planes_and_polygons_optimized(tri_mesh, avg_peaks_mat)
    else:
        all_planes, all_polygons = pl.extract_planes_and_polygons(tri_mesh, avg_peaks_mat)

    t1 = time.perf_counter()

    polylidar_time = (t1 - t0) * 1000
    logger.info("Polygon Extraction - Took (ms): %.2f", polylidar_time)
    all_poly_lines = []
    if filter_polygons:
        vertices = np.asarray(tri_mesh.vertices)
        for i in range(avg_peaks.shape[0]):
            avg_peak = avg_peaks[i, :]
            rm, _ = R.align_vectors([[0, 0, 1]], [avg_peak])
            polygons_for_normal = all_polygons[i]
            # print(polygons_for_normal)
            if len(polygons_for_normal) > 0:
                poly_lines, _ = filter_and_create_open3d_polygons(vertices, polygons_for_normal, rm=rm, **kwargs)
                all_poly_lines.extend(poly_lines)

    timings = dict(polylidar=polylidar_time)

    return all_planes, all_polygons, all_poly_lines, timings


def convert_planes_to_classified_point_cloud(all_planes, tri_mesh, all_normals, filter_kwargs=dict()):
    logger.info("Total number of normals identified: %d", len(all_planes))
    all_classified_planes = []

    # classified_plane = dict(triangles=None, points=None, class_id=None, normal=None)
    plane_counter = 0
    vertices = np.asarray(tri_mesh.vertices)
    triangles = np.asarray(tri_mesh.triangles)
    for i, normal_planes in enumerate(all_planes):
        logger.debug("Number of Planes in normal: %d", len(normal_planes))
        normal = all_normals[i, :]
        for plane in normal_planes:
            logger.debug("Number of triangles in plane: %d", len(plane))
            plane_triangles = np.asarray(plane)
            point_indices = np.unique(triangles[plane_triangles, :].flatten())
            points = np.ascontiguousarray(vertices[point_indices, :])
            normal = np.copy(normal)
            classified_plane = dict(triangles=plane_triangles, point_indices=point_indices,
                                    points=points, class_id=plane_counter, normal=normal)
            all_classified_planes.append(classified_plane)
            plane_counter += 1
    logger.info("A total of %d planes are captured", len(all_classified_planes))
    return all_classified_planes


def convert_polygons_to_classified_point_cloud(all_polygons, tri_mesh, all_normals, gt_image=None, stride=2, filter_kwargs=dict()):
    logger.info("Total number of normals identified: %d", len(all_polygons))
    all_classified_planes = []
    class_index_counter = 1  # start class indices at 1

    # The ground truth ORIGINAL image size
    window_size_out = gt_image.shape[0:2]
    # Our downsampled window size used
    window_size_in = (np.array(window_size_out) / stride).astype(np.int)
    # used to hold rasterization of polygons
    image_out = np.zeros(window_size_out, dtype=np.uint8)

    total_points = gt_image.shape[0] * gt_image.shape[1]
    # A flattened array of the point indices
    vertices = np.ascontiguousarray(gt_image[:, :, :3].reshape((total_points, 3)))
    gt_class = gt_image[:, :, 3].astype(np.uint8)

    for i, normal_polygons in enumerate(all_polygons):
        logger.debug("Number of Polygons for normal: %d", len(normal_polygons))
        normal = all_normals[i, :]
        for polygon in normal_polygons:
            polygon_shapely = convert_to_shapely_geometry_in_image_space(polygon, window_size_in, stride)
            polygon_shapely = polygon_shapely.buffer(3.0)
            polygon_shapely.simplify(1.0)
            rasterize_polygon(polygon_shapely, class_index_counter, image_out)
            point_indices = extract_image_coordinates(image_out, class_index_counter)
            points = np.ascontiguousarray(vertices[point_indices, :])
            plane_triangles = None
            normal = np.copy(normal)
            # f, (ax1, ax2) = plt.subplots(1,2)
            # ax1.imshow(image_out)
            # ax2.imshow(gt_class)
            # plt.show()
            classified_plane = dict(triangles=None, point_indices=point_indices,
                                    points=points, class_id=class_index_counter, normal=normal)
            all_classified_planes.append(classified_plane)
            class_index_counter += 1
    logger.info("A total of %d planes are captured", len(all_classified_planes))
    return all_classified_planes


def paint_planes(all_planes, auxiliary, tri_mesh_o3d):
    colors = auxiliary['map_gt_id_to_ms']
    all_colors = cc.cm.glasbey_bw(colors)[:, :3]
    all_triangles = [plane['triangles'] for plane in all_planes]

    new_mesh = assign_some_vertex_colors(tri_mesh_o3d, all_triangles, all_colors)
    return new_mesh


def get_image_peaks(ico_chart, ga, level=2, with_o3d=True,
                    find_peaks_kwargs=dict(threshold_abs=3, min_distance=1, exclude_border=False, indices=False),
                    cluster_kwargs=dict(t=0.10, criterion='distance'),
                    average_filter=dict(min_total_weight=0.01),
                    **kwargs):

    normalized_bucket_counts_by_vertex = ga.get_normalized_bucket_counts_by_vertex(True)

    ico_chart.fill_image(normalized_bucket_counts_by_vertex)
    # image = np.asarray(ico_chart.image)
    # plt.imshow(image)
    # plt.show()
    # find_peaks_kwargs = dict(threshold_abs=2, min_distance=1, exclude_border=False, indices=False)
    # cluster_kwargs = dict(t=0.10, criterion='distance')
    # average_filter = dict(min_total_weight=0.01)

    t1 = time.perf_counter()
    peaks, clusters, avg_peaks, avg_weights = find_peaks_from_ico_charts(ico_chart, np.asarray(
        normalized_bucket_counts_by_vertex), find_peaks_kwargs, cluster_kwargs, average_filter)
    t2 = time.perf_counter()
    gaussian_normals_sorted = np.asarray(ico_chart.sphere_mesh.vertices)
    # Create Open3D structures for visualization
    if with_o3d:
        pcd_all_peaks = get_pc_all_peaks(peaks, clusters, gaussian_normals_sorted)
        arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)
    else:
        pcd_all_peaks = None
        arrow_avg_peaks = None

    elapsed_time = (t2 - t1) * 1000
    timings = dict(fastga_peak=elapsed_time)

    logger.info("Peak Detection - Took (ms): %.2f", (t2 - t1) * 1000)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks, timings


def down_sample_normals(triangle_normals, down_sample_fraction=0.12, min_samples=10000, flip_normals=False, **kwargs):
    num_normals = triangle_normals.shape[0]
    to_sample = int(down_sample_fraction * num_normals)
    to_sample = max(min([num_normals, min_samples]), to_sample)
    ds_step = int(num_normals / to_sample)
    triangle_normals_ds = np.ascontiguousarray(triangle_normals[:num_normals:ds_step, :])
    if flip_normals:
        triangle_normals_ds = triangle_normals_ds * -1.0
    return triangle_normals_ds


def extract_all_dominant_plane_normals(tri_mesh, level=5, with_o3d=True, ga_=None, ico_chart_=None, **kwargs):

    # Reuse objects if provided
    if ga_ is not None:
        ga = ga_
    else:
        ga = GaussianAccumulatorS2(level=level)

    if ico_chart_ is not None:
        ico_chart = ico_chart_
    else:
        ico_chart = IcoCharts(level=level)

    triangle_normals = np.asarray(tri_mesh.triangle_normals)
    triangle_normals_ds = down_sample_normals(triangle_normals, **kwargs)

    triangle_normals_ds_mat = MatX3d(triangle_normals_ds)
    t1 = time.perf_counter()
    ga.integrate(triangle_normals_ds_mat)
    t2 = time.perf_counter()

    logger.info("Gaussian Accumulator - Normals Sampled: %d; Took (ms): %.2f",
                triangle_normals_ds.shape[0], (t2 - t1) * 1000)

    avg_peaks, pcd_all_peaks, arrow_avg_peaks, timings_dict = get_image_peaks(
        ico_chart, ga, level=level, with_o3d=with_o3d, **kwargs)

    gaussian_normals = np.asarray(ga.get_bucket_normals())
    accumulator_counts = np.asarray(ga.get_normalized_bucket_counts())

    # Create Open3D structures for visualization
    if with_o3d:
        # Visualize the Sphere
        refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
        color_counts = get_colors(accumulator_counts)[:, :3]
        colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts)
    else:
        colored_icosahedron = None

    elapsed_time_fastga = (t2 - t1) * 1000
    elapsed_time_peak = timings_dict['fastga_peak']
    elapsed_time_total = elapsed_time_fastga + elapsed_time_peak

    timings = dict(fastga_total=elapsed_time_total, fastga_integrate=elapsed_time_fastga, fastga_peak=elapsed_time_peak)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, timings


def load_pcd_file(fpath, stride=2):
    pc = pypcd.PointCloud.from_path(fpath)
    x = pc.pc_data['x']
    y = pc.pc_data['y']
    z = pc.pc_data['z']
    i = pc.pc_data['intensity']

    width = int(pc.get_metadata()['width'])
    height = int(pc.get_metadata()['height'])

    # Flat Point Cloud
    pc_np = np.column_stack((x, y, z, i))
    # Image Point Cloud (organized)
    pc_np_image = np.reshape(pc_np, (width, height, 4))
    # I'm expecting the "image" to have the rows/y-axis going down
    pc_np_image = np.ascontiguousarray(np.flip(pc_np_image, 1))

    if stride is not None:
        pc_np_image = pc_np_image[::stride, ::stride, :]
        # pc_np_image = pc_np_image[:249, :100, :]
        total_points_ds = pc_np_image.shape[0] * pc_np_image.shape[1]
        pc_np = np.reshape(pc_np_image, (total_points_ds, 4))

    pc_np_image = pc_np_image.astype(np.float64)
    pc_np = pc_np.astype(np.float64)

    return pc_np, pc_np_image
