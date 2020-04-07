import json
from pathlib import Path
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from polylidar_plane_benchmark import (DEFAULT_PPB_FILE, DEFAULT_PPB_FILE_SECONDARY, logger, SYNPEB_ALL_FNAMES, SYNPEB_DIR, SYNPEB_MESHES_DIR,
                                       SYNPEB_DIR_TEST_GT, SYNPEB_DIR_TRAIN_GT, SYNPEB_DIR_TEST_ALL, SYNPEB_DIR_TRAIN_ALL)
from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten
from polylidar_plane_benchmark.utility.helper import (load_pcd_file, create_mesh_from_organized_point_cloud,
                                                      extract_all_dominant_plane_normals, create_meshes, load_pcd_and_meshes,
                                                      extract_planes_and_polygons_from_mesh, tab40, create_open_3d_pcd)


import click


@click.group()
def visualize():
    """Visualize Data"""
    pass


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def pcd(input_file: str, stride, loops, llambda):
    """Visualize PCD File"""
    pc_raw, pc_image = load_pcd_file(input_file, stride)

    # Get just the points, no intensity
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    cmap = tab40()
    pcd_raw = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3], cmap=cmap)
    plot_meshes([pcd_raw])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def mesh(input_file: str, stride, loops, llambda):
    """Visualize Mesh from PCD File"""
    pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(input_file, stride, loops, llambda)

    # Write Smoothed Mesh to File, debugging purposes
    output_file = str(input_file).replace(str(SYNPEB_DIR), str(SYNPEB_MESHES_DIR))
    output_file = output_file.replace('.pcd', '_loops={}.ply'.format(loops))
    parent_dir = Path(output_file).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(output_file, tri_mesh_o3d)

    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def ga(input_file, stride, loops, llambda):
    """Visualize Gaussian Accumulator File"""
    pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(input_file, stride, loops, llambda)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron = extract_all_dominant_plane_normals(tri_mesh)

    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([colored_icosahedron, pcd_all_peaks, *arrow_avg_peaks], [pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def polygons(input_file, stride, loops, llambda):
    """Visualize Polygon Extraction File"""
    pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(input_file, stride, loops, llambda)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron = extract_all_dominant_plane_normals(tri_mesh)
    _, _, all_poly_lines = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks)
    mesh_3d_polylidar = []
    mesh_3d_polylidar.extend(flatten([line_mesh.cylinder_segments for line_mesh in all_poly_lines]))
    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([pcd_raw, tri_mesh_o3d, *mesh_3d_polylidar])


@visualize.command()
@click.option('-v', '--variance', type=click.Choice(['0', '1', '2', '3', '4']), default='1')
@click.option('-d', '--data', default="train")
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=10)
@click.option('--llambda', type=float, default=1.0)
@click.pass_context
def polygons_all(ctx, variance, data, stride, loops, llambda):
    """Visualize Polygon Extraction from training/testing/gt set"""
    if int(variance) == 0:
        base_dir = SYNPEB_DIR_TRAIN_GT if data == "train" else SYNPEB_DIR_TEST_GT
    else:
        base_dir = SYNPEB_DIR_TRAIN_ALL[int(
            variance) - 1] if data == "train" else SYNPEB_DIR_TEST_ALL[int(variance) - 1]

    all_fnames = SYNPEB_ALL_FNAMES
    if int(variance) != 0:
        all_fnames = all_fnames[0:10]

    for fname in all_fnames:
        fpath = str(base_dir / fname)
        logger.info("File: %s; stride=%d, loops=%d", fpath, stride, loops)
        ctx.invoke(polygons, input_file=fpath, stride=stride, loops=loops, llambda=llambda)

