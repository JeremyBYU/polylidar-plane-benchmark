import random
import time
import warnings
import gc
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

from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, open_3d_mesh_to_tri_mesh, assign_vertex_colors, get_colors
from polylidar_plane_benchmark import logger

from fastga import GaussianAccumulatorS2, MatX3d, IcoCharts
from fastga.peak_and_cluster import find_peaks_from_ico_charts
from fastga.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals

import organizedpointfilters as opf
import organizedpointfilters.cuda as opf_cuda
from organizedpointfilters import Matrix3f, Matrix3fRef


# Set the random seeds for determinism
random.seed(0)
np.random.seed(0)


def tab40():
    """A discrete colormap with 40 unique colors"""
    colors_ = np.vstack([plt.cm.tab20c.colors, plt.cm.tab20b.colors])
    return colors.ListedColormap(colors_)


def load_pcd_and_meshes(input_file, stride=2, loops=5, _lambda=0.5):
    """Load PCD and Meshes
    """
    pc_raw, pc_image = load_pcd_file(input_file, stride)
    logger.info("Visualizing Point Cloud - Size: %dX%d ; # Points: %d",
                pc_image.shape[0], pc_image.shape[1], pc_raw.shape[0])

    # Get just the points, no intensity
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    cmap = tab40()
    pcd_raw = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3], cmap=cmap)

    # tri_mesh, tri_mesh_o3d = create_meshes(pc_points, stride=stride, loops=loops)
    tri_mesh, tri_mesh_o3d = create_meshes_cuda(pc_image, loops=loops, _lambda=_lambda)

    return pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d


def filter_and_create_open3d_polygons(points, polygons, rm=None, line_radius=0.005):
    " Apply polygon filtering algorithm, return Open3D Mesh Lines "
    config_pp = dict(filter=dict(hole_area=dict(min=0.025, max=100.0), hole_vertices=dict(min=6), plane_area=dict(min=0.05)),
                     positive_buffer=0.00, negative_buffer=0.00, simplify=0.02)
    # config_pp = dict(filter=dict(hole_area=dict(min=0.00, max=100.0), hole_vertices=dict(min=6), plane_area=dict(min=0.0001)),
    #                  positive_buffer=0.00, negative_buffer=0.0, simplify=0.00)
    t1 = time.perf_counter()
    planes, obstacles = filter_planes_and_holes(polygons, points, config_pp, rm=rm)
    t2 = time.perf_counter()
    logger.info("Plane Filtering Took (ms): %.2f", (t2 - t1) * 1000)
    all_poly_lines = create_lines(planes, obstacles, line_radius=line_radius)
    return all_poly_lines, (t2 - t1) * 1000


def extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks,
                                          polylidar_kwargs=dict(alpha=0.0, lmax=0.20, min_triangles=100,
                                                                z_thresh=0.02, norm_thresh=0.98, norm_thresh_min=0.98, min_hole_vertices=6, task_threads=4)):

    pl = Polylidar3D(**polylidar_kwargs)
    avg_peaks_mat = MatrixDouble(avg_peaks)
    t0 = time.perf_counter()
    all_planes, all_polygons = pl.extract_planes_and_polygons_optimized(tri_mesh, avg_peaks_mat)
    t1 = time.perf_counter()

    polylidar_time = (t1 - t0) * 1000
    logger.info("Polygon Extraction - Took (ms): %.2f", polylidar_time)
    vertices = np.asarray(tri_mesh.vertices)
    all_poly_lines = []
    for i in range(avg_peaks.shape[0]):
        avg_peak = avg_peaks[i, :]
        rm, _ = R.align_vectors([[0, 0, 1]], [avg_peak])
        polygons_for_normal = all_polygons[i]
        # print(polygons_for_normal)
        if len(polygons_for_normal) > 0:
            poly_lines, _ = filter_and_create_open3d_polygons(vertices, polygons_for_normal, rm=rm)
            all_poly_lines.extend(poly_lines)

    return all_planes, all_polygons, all_poly_lines


def get_image_peaks(ico_chart, ga, level=2, **kwargs):

    normalized_bucket_counts_by_vertex = ga.get_normalized_bucket_counts_by_vertex(True)

    ico_chart.fill_image(normalized_bucket_counts_by_vertex)
    # image = np.asarray(ico_chart.image)
    # plt.imshow(image)
    # plt.show()
    find_peaks_kwargs = dict(threshold_abs=2, min_distance=1, exclude_border=False, indices=False)
    cluster_kwargs = dict(t=0.10, criterion='distance')
    average_filter = dict(min_total_weight=0.01)

    t1 = time.perf_counter()
    peaks, clusters, avg_peaks, avg_weights = find_peaks_from_ico_charts(ico_chart, np.asarray(
        normalized_bucket_counts_by_vertex), find_peaks_kwargs, cluster_kwargs, average_filter)
    t2 = time.perf_counter()
    gaussian_normals_sorted = np.asarray(ico_chart.sphere_mesh.vertices)
    pcd_all_peaks = get_pc_all_peaks(peaks, clusters, gaussian_normals_sorted)
    arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)

    logger.info("Peak Detection - Took (ms): %.2f", (t2 - t1) * 1000)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks


def down_sample_normals(triangle_normals, ds=4, min_samples=10000, flip_normals=False, **kwargs):
    num_normals = triangle_normals.shape[0]
    ds_normals = int(num_normals / ds)
    to_sample = max(min([num_normals, min_samples]), ds_normals)
    ds_step = int(num_normals / to_sample)
    triangle_normals_ds = np.ascontiguousarray(triangle_normals[:num_normals:ds_step, :])
    if flip_normals:
        triangle_normals_ds = triangle_normals_ds * -1.0
    return triangle_normals_ds


def extract_all_dominant_plane_normals(tri_mesh, level=5, **kwargs):
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

    logger.info("Gaussian Accumulator - Normals Sampled: %d; Took (ms): %.2f",
                triangle_normals_ds.shape[0], (t2 - t1) * 1000)

    # Visualize the Sphere
    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
    color_counts = get_colors(accumulator_counts)[:, :3]
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts)

    # 2D peak detection
    avg_peaks, pcd_all_peaks, arrow_avg_peaks = get_image_peaks(ico_chart, ga, level=level)

    return avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron


def create_meshes(pc_points, stride=2, loops=5, _lambda=0.5,):
    tri_mesh = create_mesh_from_organized_point_cloud(pc_points, stride=stride)

    tri_mesh_o3d = create_open_3d_mesh(np.asarray(tri_mesh.triangles), pc_points)
    # Perform Smoothing
    filter_scope = o3d.geometry.FilterScope(0)
    kwargs = dict(filter_scope=filter_scope)
    kwargs['lambda'] = _lambda
    t1 = time.perf_counter()
    if loops > 0:
        tri_mesh_o3d = tri_mesh_o3d.filter_smooth_laplacian(loops)
    t2 = time.perf_counter()
    tri_mesh_o3d = tri_mesh_o3d.compute_triangle_normals()

    tri_mesh = open_3d_mesh_to_tri_mesh(tri_mesh_o3d)

    logger.info("Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    return tri_mesh, tri_mesh_o3d


def create_mesh_from_organized_point_cloud_with_o3d(pcd: np.ndarray, rows=500, cols=500, stride=2):
    pcd_ = pcd
    if pcd.ndim == 3:
        rows = pcd.shape[0]
        cols = pcd.shape[1]
        stride = 1
        pcd_ = pcd.reshape((rows * cols, 3))

    pcd_mat = MatrixDouble(pcd_, copy=True)
    pcd_mat_np = np.asarray(pcd_mat)
    tri_mesh = extract_tri_mesh_from_organized_point_cloud(pcd_mat, rows, cols, stride)

    tri_mesh_o3d = create_open_3d_mesh(np.asarray(tri_mesh.triangles), pcd_)

    return tri_mesh, tri_mesh_o3d


def laplacian_opc(opc, loops=5, _lambda=0.5, kernel_size=3):
    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    a_ref = Matrix3fRef(opc_float)

    t1 = time.perf_counter()
    b_cp = opf.kernel.laplacian_K3(a_ref, _lambda=_lambda, iterations=loops)
    t2 = time.perf_counter()
    logger.info("OPC Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    opc_float_out = np.asarray(b_cp)
    opc_out = opc_float_out.astype(np.float64)

    return opc_out


def laplacian_opc_cuda(opc, loops=5, _lambda=0.5, **kwargs):

    opc_float = (np.ascontiguousarray(opc[:, :, :3])).astype(np.float32)

    t1 = time.perf_counter()
    opc_float_out = opf_cuda.kernel.laplacian_K3_cuda(opc_float, loops=loops, _lambda=_lambda, **kwargs)
    t2 = time.perf_counter()

    logger.info("OPC CUDA Laplacian Mesh Smoothing Took (ms): %.2f", (t2 - t1) * 1000)

    # only for visualization purposes here
    opc_out = opc_float_out.astype(np.float64)

    return opc_out


def create_meshes_cuda(opc, loops=5, _lambda=0.5):
    """Creates a mesh from a noisy organized point cloud

    Arguments:
        opc {ndarray} -- Must be MXNX3

    Keyword Arguments:
        loops {int} -- How many loop iterations (default: {5})
        _lambda {float} -- weighted iteration movement (default: {0.5})

    Returns:
        [tuple(mesh, o3d_mesh)] -- polylidar mesh and o3d mesh reperesentation
    """
    smooth_opc = laplacian_opc_cuda(opc, loops=loops, _lambda=_lambda)
    tri_mesh, tri_mesh_o3d = create_mesh_from_organized_point_cloud_with_o3d(smooth_opc)

    return tri_mesh, tri_mesh_o3d


def create_mesh_from_organized_point_cloud(pcd, rows=500, cols=500, stride=2):
    pcd_mat = MatrixDouble(pcd)
    pcd_mat_np = np.asarray(pcd_mat)
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
    # I'm expecting the "image" to have the rows/y-axis going down
    pc_np_image = np.ascontiguousarray(np.flip(pc_np_image, 0))
    if stride is not None:
        pc_np_image = pc_np_image[::stride, ::stride, :]
        total_points_ds = pc_np_image.shape[0] * pc_np_image.shape[1]
        pc_np = np.reshape(pc_np_image, (total_points_ds, 4))

    pc_np_image = pc_np_image.astype(np.float64)
    pc_np = pc_np.astype(np.float64)

    return pc_np, pc_np_image
