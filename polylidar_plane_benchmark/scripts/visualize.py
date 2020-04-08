import json
from pathlib import Path
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import colorcet as cc

from polylidar_plane_benchmark import (DEFAULT_PPB_FILE, DEFAULT_PPB_FILE_SECONDARY, logger, SYNPEB_ALL_FNAMES, SYNPEB_DIR, SYNPEB_MESHES_DIR,
                                       SYNPEB_DIR_TEST_GT, SYNPEB_DIR_TRAIN_GT, SYNPEB_DIR_TEST_ALL, SYNPEB_DIR_TRAIN_ALL)
from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten, mark_invalid_planes
from polylidar_plane_benchmark.utility.helper import (load_pcd_file, create_mesh_from_organized_point_cloud, convert_planes_to_classified_point_cloud,
                                                      extract_all_dominant_plane_normals, create_meshes, load_pcd_and_meshes,
                                                      extract_planes_and_polygons_from_mesh, create_open_3d_pcd, paint_planes)
from polylidar_plane_benchmark.utility.evaluate import evaluate


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

    # pc_raw_filt = pc_raw[pc_raw[:, 3] == 3.0, :]
    # Get just the points, no intensity
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # Create Open3D point cloud
    cmap = cc.cm.glasbey_bw
    pcd_raw = create_open_3d_pcd(pc_points[:, :3], pc_raw[:, 3], cmap=cmap)
    plot_meshes([pcd_raw])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def mesh(input_file: str, stride, loops, llambda):
    """Visualize Mesh from PCD File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, _ = load_pcd_and_meshes(input_file, stride, loops, llambda)

    # Write Smoothed Mesh to File, debugging purposes
    output_file = str(input_file).replace(str(SYNPEB_DIR), str(SYNPEB_MESHES_DIR))
    output_file = output_file.replace('.pcd', '_loops={}.ply'.format(loops))
    parent_dir = Path(output_file).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(output_file, tri_mesh_o3d)

    plot_meshes([pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def ga(input_file, stride, loops, llambda):
    """Visualize Gaussian Accumulator File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, _ = load_pcd_and_meshes(input_file, stride, loops, llambda)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, _ = extract_all_dominant_plane_normals(tri_mesh)

    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([colored_icosahedron, pcd_all_peaks, *arrow_avg_peaks], [pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def polygons(input_file, stride, loops, llambda):
    """Visualize Polygon Extraction File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, _ = load_pcd_and_meshes(input_file, stride, loops, llambda)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, _ = extract_all_dominant_plane_normals(tri_mesh)
    _, _, all_poly_lines, _ = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks)
    mesh_3d_polylidar = []
    mesh_3d_polylidar.extend(flatten([line_mesh.cylinder_segments for line_mesh in all_poly_lines]))
    
    plot_meshes([pcd_raw, tri_mesh_o3d, *mesh_3d_polylidar])




@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=20)
@click.option('--llambda', type=float, default=1.0)
def planes(input_file, stride, loops, llambda):
    """Visualize Polygon Extraction File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, mesh_timings = load_pcd_and_meshes(input_file, stride, loops, llambda)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, fastga_timings = extract_all_dominant_plane_normals(
        tri_mesh)
    all_planes, _,  _, polylidar_timings = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks, filter_polygons=False)

    all_timings = dict(**mesh_timings, **fastga_timings, **polylidar_timings)
    # print(all_timings)
    # def convert_planes_to_classified_point_cloud(all_planes, tri_mesh, all_normals, filter_kwargs=dict()):
    all_planes_classified = convert_planes_to_classified_point_cloud(all_planes, tri_mesh, avg_peaks)
    results, auxiliary = evaluate(pc_image, all_planes_classified)
    # all_planes_classified = all_planes_classified[34:35]
    tri_mesh_o3d_painted = paint_planes(all_planes_classified, tri_mesh_o3d)

    invalid_plane_markers = mark_invalid_planes(pc_raw, auxiliary, all_planes_classified)
    invalid_plane_markers = []
    # Get just the points, no intensity

    # Create Open3D point cloud
    # pc_raw_filt = pc_raw[pc_raw[:, 3] == 22.0, :]
    # pc_points = np.ascontiguousarray(pc_raw_filt[:, :3])
    # pcd_raw = create_open_3d_pcd(pc_points[:, :3])
    # pcd_raw.paint_uniform_color([0, 1, 0])

    # bad_indices = np.array([ 7768,  8018,  8019,  8269,  8516,  8520,  8770,  9021,  9271,  9517,  9522,  9767,  9772, 10023, 10273, 10523, 10524, 10774, 11025, 11275, 11519, 11776, 12277, 12528, 12778, 13029, 13279, 13529, 13780, 14030, 14281, 14782, 15283, 15533,
    #    15783, 15784, 16034, 16284, 16285, 16535, 16786, 17036, 17286, 17287, 17537, 17787, 17788, 18038, 18288, 18539, 18789, 19040, 19290, 19540, 19541, 19791, 20041, 20042, 20292, 20542, 20543, 20793, 21043, 21044, 21294, 21544, 21545, 21794,
    #    21795, 22045])
    # bad_indices = bad_indices.astype(np.int)
    # pc_bad = pc_raw[bad_indices, :]
    # pcd_bad = create_open_3d_pcd(pc_bad[:, :3])
    # pcd_bad.paint_uniform_color([0.5, 0.5, 1.0])


    # plot_meshes([pcd_raw, pcd_bad, tri_mesh_o3d_painted])
    plot_meshes([pcd_raw, tri_mesh_o3d_painted, *invalid_plane_markers])


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


def main():
    visualize()

if __name__ == "__main__":
    main()