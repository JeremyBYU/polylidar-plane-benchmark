import json
from pathlib import Path
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import colorcet as cc
import pandas as pd
import seaborn as sns

from polylidar_plane_benchmark import (DEFAULT_PPB_FILE, DEFAULT_PPB_FILE_SECONDARY, logger, SYNPEB_ALL_FNAMES, SYNPEB_DIR, SYNPEB_MESHES_DIR,
                                       SYNPEB_DIR_TEST_GT, SYNPEB_DIR_TRAIN_GT, SYNPEB_DIR_TEST_ALL, SYNPEB_DIR_TRAIN_ALL)
from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten, mark_invalid_planes
from polylidar_plane_benchmark.utility.helper import (load_pcd_file, convert_planes_to_classified_point_cloud,
                                                      extract_all_dominant_plane_normals, load_pcd_and_meshes, convert_polygons_to_classified_point_cloud,
                                                      extract_planes_and_polygons_from_mesh, create_open_3d_pcd, paint_planes)
from polylidar_plane_benchmark.utility.evaluate import evaluate


import click


# Variance 1 - Laplacian Smoothing Loops 4

@click.group()
def visualize():
    """Visualize Data"""
    pass


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
def pcd(input_file: str, stride):
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
@click.option('-l', '--loops', type=int, default=5)
@click.option('--llambda', type=float, default=1.0)
@click.option('-ks', '--kernel-size', type=int, default=3)
@click.option('-lb', '--loops-bilateral', type=int, default=0)
def mesh(input_file: str, stride, loops, llambda, kernel_size, loops_bilateral):
    """Visualize Mesh from PCD File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, _ = load_pcd_and_meshes(
        input_file, stride, loops, llambda, loops_bilateral, kernel_size=kernel_size)

    # Write Smoothed Mesh to File, debugging purposes
    output_file = str(input_file).replace(str(SYNPEB_DIR), str(SYNPEB_MESHES_DIR))
    output_file = output_file.replace('.pcd', '_loops={}.ply'.format(loops))
    parent_dir = Path(output_file).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(output_file, tri_mesh_o3d)

    # import ipdb; ipdb.set_trace()
    # mask = pc_raw[:, 3] == 3.0
    # colors = np.asarray(pcd_raw.colors)
    # colors[mask] = [0.0,1.0,0]

    plot_meshes([pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=5)
@click.option('--llambda', type=float, default=1.0)
@click.option('-ks', '--kernel-size', type=int, default=3)
@click.option('-lb', '--loops-bilateral', type=int, default=0)
def ga(input_file, stride, loops, llambda, kernel_size, loops_bilateral):
    """Visualize Gaussian Accumulator File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, _ = load_pcd_and_meshes(
        input_file, stride, loops, llambda, loops_bilateral, kernel_size=kernel_size)

    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, _ = extract_all_dominant_plane_normals(tri_mesh)

    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([colored_icosahedron, pcd_all_peaks, *arrow_avg_peaks], [pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=5)
@click.option('--llambda', type=float, default=1.0)
@click.option('-ks', '--kernel-size', type=int, default=3)
@click.option('-lb', '--loops-bilateral', type=int, default=0)
def polygons(input_file, stride, loops, llambda, kernel_size, loops_bilateral):
    """Visualize Polygon Extraction File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, _ = load_pcd_and_meshes(
        input_file, stride, loops, llambda, loops_bilateral, kernel_size=kernel_size)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, _ = extract_all_dominant_plane_normals(tri_mesh)
    _, _, all_poly_lines, _ = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks)
    mesh_3d_polylidar = []
    mesh_3d_polylidar.extend(flatten([line_mesh.cylinder_segments for line_mesh in all_poly_lines]))

    plot_meshes([pcd_raw, tri_mesh_o3d, *mesh_3d_polylidar])

def plot_triangle_normals(normals: np.ndarray):
    colors = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    im = colors.reshape((249, 249, 2, 3))
    im = im[:, :, 1, :]
    plt.imshow(im, origin='upper')
    plt.show()


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=5)
@click.option('--llambda', type=float, default=1.0)
@click.option('-ks', '--kernel-size', type=int, default=3)
@click.option('-lb', '--loops-bilateral', type=int, default=0)
def planes(input_file, stride, loops, llambda, kernel_size, loops_bilateral):
    """Visualize Polygon Extraction File"""
    pc_raw, pcd_raw, pc_image, tri_mesh, tri_mesh_o3d, mesh_timings = load_pcd_and_meshes(
        input_file, stride, loops, llambda, loops_bilateral, kernel_size=kernel_size)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron, fastga_timings = extract_all_dominant_plane_normals(
        tri_mesh)

    # print(avg_peaks)
    # print((avg_peaks * 0.5 + 0.5) * 255)

    # tri_mesh_normals = np.asarray(tri_mesh.triangle_normals)
    # plot_triangle_normals(tri_mesh_normals)

    all_planes, all_polygons, _, polylidar_timings = extract_planes_and_polygons_from_mesh(
        tri_mesh, avg_peaks, filter_polygons=False)

    all_timings = dict(**mesh_timings, **fastga_timings, **polylidar_timings)
    all_planes_classified = convert_planes_to_classified_point_cloud(all_planes, tri_mesh, avg_peaks)
    # paint the planes
    # all_planes_classified.append(dict(triangles=np.array([51032])))
    tri_mesh_o3d_painted = paint_planes(all_planes_classified, tri_mesh_o3d)
    # del all_planes_classified[-1]

    # can be evaluated by polygons (using downsampled image) or just the planes
    # for evaluation we need the full point cloud, not downsampled
    # _, gt_image = load_pcd_file(input_file, stride=1)
    # all_planes_classified = convert_polygons_to_classified_point_cloud(all_polygons, tri_mesh, avg_peaks, gt_image, stride,)
    # results, auxiliary = evaluate(gt_image, all_planes_classified)
    # get results
    results, auxiliary = evaluate(pc_image, all_planes_classified, tcomp=0.8)

    # create invalid plane markers, green = gt_label_missed, red=ms_labels_noise, blue=gt_label_over_seg,gray=ms_label_under_seg
    invalid_plane_markers = mark_invalid_planes(pc_raw, auxiliary, all_planes_classified)

    # invalid_plane_markers = []
    # plot_meshes([tri_mesh_o3d_painted])
    plot_meshes([pcd_raw, tri_mesh_o3d_painted, *invalid_plane_markers])


@visualize.command()
@click.option('-v', '--variance', type=click.Choice(['0', '1', '2', '3', '4']), default='1')
@click.option('-d', '--data', default="train")
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=5)
@click.option('--llambda', type=float, default=1.0)
@click.option('-ks', '--kernel-size', type=int, default=3)
@click.option('-lb', '--loops-bilateral', type=int, default=0)
@click.pass_context
def planes_all(ctx, variance, data, stride, loops, llambda, kernel_size, loops_bilateral):
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
        ctx.invoke(planes, input_file=fpath, stride=stride, loops=loops, llambda=llambda,
                   kernel_size=kernel_size, loops_bilateral=loops_bilateral)


@visualize.command()
@click.option('-v', '--variance', type=click.Choice(['0', '1', '2', '3', '4']), default='1')
@click.option('-d', '--data', default="train")
@click.option('-s', '--stride', type=int, default=2)
@click.option('-l', '--loops', type=int, default=5)
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
