import json
from pathlib import Path
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from polylidar_plane_benchmark import DEFAULT_PPB_FILE, DEFAULT_PPB_FILE_SECONDARY, logger
from polylidar_plane_benchmark.utility.o3d_util import create_open_3d_pcd, plot_meshes, get_arrow, create_open_3d_mesh, flatten
from polylidar_plane_benchmark.utility.helper import (load_pcd_file, create_mesh_from_organized_point_cloud,
                                                      extract_all_dominant_plane_normals, create_meshes, load_pcd_and_meshes,
                                                      extract_planes_and_polygons_from_mesh)
                            

import click


@click.group()
def visualize():
    """Visualize Data"""
    pass


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=1)
@click.option('-l', '--loops', type=int, default=5)
def pcd(input_file, stride, loops):
    """Visualize PCD File"""
    pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(input_file, stride, loops)

    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=1)
@click.option('-l', '--loops', type=int, default=5)
def ga(input_file, stride, loops):
    """Visualize PCD File"""
    pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(input_file, stride, loops)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron = extract_all_dominant_plane_normals(tri_mesh)

    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([colored_icosahedron, pcd_all_peaks, *arrow_avg_peaks], [pcd_raw, tri_mesh_o3d])


@visualize.command()
@click.option('-i', '--input-file', type=click.Path(exists=True), default=DEFAULT_PPB_FILE)
@click.option('-s', '--stride', type=int, default=1)
@click.option('-l', '--loops', type=int, default=5)
def polygons(input_file, stride, loops):
    """Visualize PCD File"""
    pc_raw, pcd_raw, tri_mesh, tri_mesh_o3d = load_pcd_and_meshes(input_file, stride, loops)
    avg_peaks, pcd_all_peaks, arrow_avg_peaks, colored_icosahedron = extract_all_dominant_plane_normals(tri_mesh)
    _, _, all_poly_lines = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks)
    mesh_3d_polylidar = []
    mesh_3d_polylidar.extend(flatten([line_mesh.cylinder_segments for line_mesh in all_poly_lines]))
    # arrow = get_arrow(origin=[0,0,0], end=[3, 0, 0], cylinder_radius=0.01)
    plot_meshes([pcd_raw, tri_mesh_o3d, *mesh_3d_polylidar])

