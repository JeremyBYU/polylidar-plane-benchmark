"""This module contains plane fitting procedures. This is only needed to *evaluate* RMSE of a plane

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

np.set_printoptions(threshold=3600, linewidth=350, precision=6, suppress=True)

# Source: https://stackoverflow.com/a/38770513/9341063


def PCA(data, correlation=False, sort=True):
    """ Applies Principal Component Analysis to the data

    Parameters
    ----------        
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix. 

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.   
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

# Source: https://stackoverflow.com/a/38770513/9341063


def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------        
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])

    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.    
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:, 2]
    normal = normal / np.linalg.norm(normal)

    #: get a point from the plane
    point = np.mean(points, axis=0)

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal


def point_to_plane_distance_simple(points, p_point, p_normal: np.array):
    distance = []
    for i in range(points.shape[0]):
        point = points[i, :]
        signed_distance = p_normal.dot(p_point - point)
        distance.append(signed_distance)
    distance = np.array(distance)
    rmse = np.sqrt(np.mean(distance ** 2))
    return distance, rmse


def calculate_rmse(points: np.array, p_point: np.array, p_normal: np.array):
    points_a = points - p_point
    distance = points_a @ p_normal
    distance = np.array(distance)
    rmse = np.sqrt(np.mean(distance ** 2))
    return distance, rmse


def fit_plane_and_get_rmse(points: np.array):
    """Fit Plane to 3D point and return RMSE

    Arguments:
        points {np.array} -- Point cloud, NX3

    Returns:
        tuple(point, normal, rmse) -- Return a for a point on the plane, its normal, and the RMSE
    """

    point, normal = best_fitting_plane(points)
    distance, rmse = calculate_rmse(points, point, normal)
    return point, normal, distance, rmse


def main():
    direction = 0.7071
    normalizing = np.sqrt(3 * direction * direction)
    unit = direction / normalizing
    a = unit
    b = unit
    c = unit
    d = 1

    noise = 1

    print(f"Ground Truth Normal: [{a:.2f},{b:.2f},{c:.2f}], d: {d}")

    # Create point cloud
    pc_gt = []
    pc_ns = []
    for x in range(-10, 10, 1):
        for y in range(-10, 10, 1):
            true_z = -(a * x + b * y) / c + d
            noise_z = true_z + np.random.rand() * noise
            pc_gt.append([x, y, true_z])
            pc_ns.append([x, y, noise_z])

    pc_gt = np.array(pc_gt)
    pc_ns = np.array(pc_ns)

    # fit plane and get rmse
    point, normal, distance, rmse = fit_plane_and_get_rmse(pc_ns)
    print(f"Predicted Normal: {normal}; point: {point}; RMSE: {rmse:.3f}")

    # plot data
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(pc_ns[:, 0], pc_ns[:, 1], pc_ns[:, 2], color='b')
    plt.show()

    # norms = np.apply_along_axis(np.linalg.norm, 1, pc_ns)
    # pc_ns_norm = pc_ns / np.expand_dims(norms, axis=1)
    # ax.scatter(pc_ns_norm[:, 0], pc_ns_norm[:, 1], pc_ns_norm[:, 2], color='g')
    # print(pc_ns)


if __name__ == "__main__":
    main()
