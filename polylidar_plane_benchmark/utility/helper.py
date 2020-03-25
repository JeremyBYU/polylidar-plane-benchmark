import random
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pypcd.pypcd as pypcd
from scipy.spatial.transform import Rotation as R

from polylidar import extract_point_cloud_from_float_depth, extract_tri_mesh_from_organized_point_cloud, MatrixDouble
from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, open_3d_mesh_to_tri_mesh, assign_vertex_colors, get_colors
from polylidar_plane_benchmark import logger

from fastga import GaussianAccumulatorS2, MatX3d, IcoCharts
from fastga.peak_and_cluster import find_peaks_from_ico_charts
from fastga.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals

import cv2

# Set the random seeds for determinism
random.seed(0)
np.random.seed(0)

def load_pcd_and_meshes(input_file, stride=2, loops=5):
    """Load PCD and Meshes
    """
    pc_raw, depth_image = load_pcd_file(input_file, stride)
    logger.info("Visualizing Point Cloud - Size: %dX%d ; # Points: %d", depth_image.shape[0], depth_image.shape[1],  pc_raw.shape[0])

    # Get just the points, no intensity
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    pcd_raw = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3])

    tri_mesh, tri_mesh_o3d = create_meshes(pc_points, stride=stride, loops=loops)

    return pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d


def get_image_peaks(ico_chart, ga, level=2, **kwargs):

    normalized_bucket_counts_by_vertex = ga.get_normalized_bucket_counts_by_vertex(True)
    ico_chart.fill_image(normalized_bucket_counts_by_vertex)
    find_peaks_kwargs=dict(threshold_abs=5, min_distance=1, exclude_border=False, indices=False)
    cluster_kwargs=dict(t=0.15, criterion='distance')
    average_filter=dict(min_total_weight=0.02)


    t1 = time.perf_counter()
    peaks, clusters, avg_peaks, avg_weights = find_peaks_from_ico_charts(ico_chart, np.asarray(normalized_bucket_counts_by_vertex), find_peaks_kwargs, cluster_kwargs, average_filter)
    t2 = time.perf_counter()
    gaussian_normals_sorted = np.asarray(ico_chart.sphere_mesh.vertices)
    pcd_all_peaks = get_pc_all_peaks(peaks, clusters, gaussian_normals_sorted)
    arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)

    logger.info("Peak Detection - Took (ms): %.2f",  (t2 - t1) * 1000)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks


def down_sample_normals(triangle_normals, ds=5, min_samples=10000, flip_normals=True, **kwargs):
    num_normals = triangle_normals.shape[0]
    ds_normals = int(num_normals / ds)
    to_sample = max(min([num_normals, min_samples]), ds_normals)
    ds_step = int(num_normals / to_sample)
    triangle_normals_ds = np.ascontiguousarray(triangle_normals[:num_normals:ds_step, :])
    if flip_normals:
        triangle_normals_ds = triangle_normals_ds * -1.0
    return triangle_normals_ds

def extract_all_dominant_planes(tri_mesh, level=5, **kwargs):
    # TODO don't create new everytime
    ga = GaussianAccumulatorS2(level=level)
    ico_chart = IcoCharts(level=level)

    triangle_normals = np.asarray(tri_mesh.triangle_normals)
    triangle_normals_ds = down_sample_normals(triangle_normals, **kwargs)

    triangle_normals_ds_mat = MatX3d(triangle_normals_ds)
    t1 = time.perf_counter()
    ga.integrate(triangle_normals_ds_mat)
    t2 = time.perf_counter()
    gaussian_normals = np.asarray(ga.get_bucket_normals())
    accumulator_counts = np.asarray(ga.get_normalized_bucket_counts())

    logger.info("Gaussian Accumulator - PC Size: %d; Took (ms): %.2f", triangle_normals_ds.shape[0], (t2 - t1) * 1000)

    # Visualize the Sphere
    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
    color_counts = get_colors(accumulator_counts)[:, :3]
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts)

    # 2D peak detection
    avg_peaks, pcd_all_peaks, arrow_avg_peaks = get_image_peaks(ico_chart, ga, level=level)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron


def create_meshes(pc_points, stride=2, loops=5):
    tri_mesh = create_mesh_from_organized_point_cloud(pc_points, stride=stride)
    tri_mesh_o3d = create_open_3d_mesh(np.asarray(tri_mesh.triangles), pc_points)

    # Perform Smoothing
    t1 = time.perf_counter()
    tri_mesh_o3d = tri_mesh_o3d.filter_smooth_laplacian(loops)
    t2 = time.perf_counter()
    tri_mesh_o3d.compute_triangle_normals()

    tri_mesh = open_3d_mesh_to_tri_mesh(tri_mesh_o3d)

    logger.info("Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    return tri_mesh, tri_mesh_o3d


def create_mesh_from_organized_point_cloud(pcd, rows=500, cols=500, stride=2):
    pcd_mat = MatrixDouble(pcd)
    tri_mesh = extract_tri_mesh_from_organized_point_cloud(pcd_mat, rows, cols, stride)
    return tri_mesh


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

    if stride is not None:
        pc_np_image = pc_np_image[::stride, ::stride, :]
        total_points_ds = pc_np_image.shape[0] * pc_np_image.shape[1]
        pc_np = np.reshape(pc_np_image, (total_points_ds, 4))

    depth_image = np.ascontiguousarray(pc_np_image[:, :, 0])
    pc_np = pc_np.astype(np.float64)

    return pc_np, depth_image

# def load_pcd_file(fpath, ds=None):
#     pc = pypcd.PointCloud.from_path(fpath)
#     x = pc.pc_data['x']
#     y = pc.pc_data['y']
#     z = pc.pc_data['z']
#     i = pc.pc_data['intensity']

#     width = int(pc.get_metadata()['width'])
#     height = int(pc.get_metadata()['height'])

#     # Flat Point Cloud
#     pc_np = np.column_stack((x, y, z, i))
#     # Image Point Cloud (organized)
#     pc_np_image = np.reshape(pc_np, (width, height, 4))
#     depth_image = np.ascontiguousarray(pc_np_image[:, :, 0])

#     if ds is not None:
#         pc_np_image = pc_np_image[::ds, ::ds, :]
#         total_points_ds = pc_np_image.shape[0] * pc_np_image.shape[1]
#         pc_np = np.reshape(pc_np_image, (total_points_ds, 4))

#     intrinsics = get_intrinsic_matrix(depth_image)
#     # intrinsics, distortion, rotation_matrix, tranlsation  = estimate_intrinsics_matrix(depth_image, pc_np_image)
#     # rm = R.from_matrix(rotation_matrix)

#     pc_from_depth = depth_to_pc(depth_image, intrinsics, ds, distortion=None, rm=None)
#     pc_from_depth_rotated = pc_from_depth

#     return pc_np, pc_from_depth_rotated, depth_image
