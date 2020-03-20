import json
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from polylidar_plane_benchmark import DEFAULT_PPB_FILE, DEFAULT_PPB_FILE_SECONDARY, logger
from polylidar_plane_benchmark.utility.helper import load_pcd_file, create_mesh_from_organized_point_cloud
from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh

import click



@click.group()
def visualize():
    """Visualize Data"""
    pass


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=1)
def pcd(input_file, stride):
    """Visualize PCD File"""
    pc_raw, depth_image = load_pcd_file(input_file, stride)
    pc_points = np.ascontiguousarray(pc_raw[:, :3])
    # import pdb; pdb.set_trace()
    tri_mesh = create_mesh_from_organized_point_cloud(pc_points, stride=stride)
    tri_mesh_o3d = create_open_3d_mesh(np.asarray(tri_mesh.triangles), pc_points)
    pcd_raw = create_open_3d_pcd(pc_raw[:, :3], pc_raw[:, 3])
    logger.info("Visualizing Point Cloud - Size: %dX%d ; # Points: %d", depth_image.shape[0], depth_image.shape[1],  pc_raw.shape[0])
    o3d.io.write_triangle_mesh("noisy_mesh_ds_4.ply", tri_mesh_o3d)
    plt.imshow(depth_image)
    plt.show()

    # pc_2 , _, _= load_pcd_file(DEFAULT_PPB_FILE_SECONDARY, ds=1)
    # pcd_2 = create_open_3d_pcd(pc_2[:, :3], pc_2[:, 3])
    # plot_meshes([pcd, pcd_2])
    arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.translate([-2.0, 0, 0])
    o3d.visualization.draw_geometries([pcd_raw, tri_mesh_o3d, arrow, axis])
    # plot_meshes([pcd_raw, pcd_depth, arrow])
