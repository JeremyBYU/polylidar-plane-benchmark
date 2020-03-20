import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pypcd.pypcd as pypcd
from polylidar import extract_point_cloud_from_float_depth
from scipy.spatial.transform import Rotation as R

import cv2

# Set the random seeds for determinism
random.seed(0)
np.random.seed(0)


def get_colors(inp, colormap=plt.cm.viridis, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


# K = [[f, 0, Cu],
# [0, f, Cv],
# [0, 0, 1 ]]

# fx=fy=f=imageWidth /(2 * tan(CameraFOV * π / 360))

# Cu=image horizontal center = imageWidth/2
# Cv=image vertical center = imageHight/2


def get_intrinsic_matrix(depth_image, fov=70):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    fov_rad = np.radians(fov)
    f = width / (2 *  np.tan(fov_rad/2.0))
    cu =  width / 2.0
    cv = height / 2.0
    intrinsics = np.array(
        [[f, 0, cu],
        [0, f, cv],
        [0, 0, 1]]
    )
    return intrinsics

def depth_to_pc(depth_image, intrinsics, ds=1, distortion=None, rm=None):
    # distortion_ = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) if distortion is None else distortion
    # distortion_[0,0] = 0.00
    # distortion_[0,1] = 0.0
    extrinsics = np.identity(4)
    # intrinsics = get_intrinsic_matrix(depth_image)
    pc = extract_point_cloud_from_float_depth(depth_image, intrinsics, extrinsics, stride=ds)
    pc = np.asarray(pc)
    total_points = int(pc.size / 3)
    pc = np.reshape(pc, (total_points, 3))

    if rm is not None:
        pc = rm.apply(pc)

    pc_temp = np.copy(pc)
    pc_temp[:, 0] = pc[:, 2]
    pc_temp[:, 1] = pc[:, 0]
    pc_temp[:, 2] = -pc[:, 1]


    return pc_temp

def estimate_intrinsics_matrix(depth_image, pc_gt):
    rows = depth_image.shape[0]
    cols = depth_image.shape[1]
    pixel_rows = np.arange(0, rows, 5)
    pixel_cols = np.arange(0, cols, 5)
    xv, yv = np.meshgrid(pixel_rows, pixel_cols)
    coordinate_grid = np.array([xv, yv]) 
    # import pdb; pdb.set_trace()
    pixels = coordinate_grid.swapaxes(0, 2)
    pixels = pixels.reshape((pixels.shape[0] * pixels.shape[1], 2))
    print(pixels)

    object_points = (pc_gt[pixels[:, 0], pixels[:, 1], :3]).astype(np.float32)
    print(object_points)
    pixels = pixels.astype(np.float32)
    pc_temp = np.copy(object_points)
    pc_temp[:, 2] = object_points[:, 0]
    pc_temp[:, 0] = -object_points[:, 1]
    pc_temp[:, 1] = -object_points[:, 2]
    # object_points = object_points.tolist()
    # pixels = pixels.tolist()
    # print(object_points)
    # print(pixels)
    intrinsics = get_intrinsic_matrix(depth_image)
    dist_coefs = np.zeros(4)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([pc_temp], [pixels], depth_image.shape[::-1], intrinsics, dist_coefs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    dist = np.ascontiguousarray(np.transpose(dist)).astype(np.float64)
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
    translation = tvecs[0]

    return intrinsics, dist, rotation_matrix, translation

def load_pcd_file(fpath, ds=None):
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
    depth_image = np.ascontiguousarray(pc_np_image[:, :, 0])

    if ds is not None:
        pc_np_image = pc_np_image[::ds, ::ds, :]
        total_points_ds = pc_np_image.shape[0] * pc_np_image.shape[1]
        pc_np = np.reshape(pc_np_image, (total_points_ds, 4))
    
    intrinsics = get_intrinsic_matrix(depth_image)
    # intrinsics, distortion, rotation_matrix, tranlsation  = estimate_intrinsics_matrix(depth_image, pc_np_image)
    # rm = R.from_matrix(rotation_matrix)

    pc_from_depth = depth_to_pc(depth_image, intrinsics, ds, distortion=None, rm=None)
    pc_from_depth_rotated = pc_from_depth

    return pc_np, pc_from_depth_rotated, depth_image

