from shapely.geometry import asPolygon, Polygon
import numpy as np
from rasterio.features import rasterize
def convert_image_indices(point_indices, window_size, stride=1, flip_xy=True):
    point_indices_np = np.asarray(point_indices)
    image_indices = np.column_stack(np.unravel_index(point_indices_np, window_size))
    image_indices = image_indices * stride
    if flip_xy:
        image_indices = np.flip(image_indices, axis=1)
    return image_indices

def convert_to_shapely_geometry_in_image_space(polygon, window_size=(250, 250), stride=2):
    assert len(window_size) == 2, "Window size must be 2 dimensional"
    shell = convert_image_indices(polygon.shell, window_size, stride)
    holes = np.array([convert_image_indices(hole, window_size, stride) for hole in polygon.holes])

    polygon = asPolygon(shell=shell, holes=holes)
    return polygon

def rasterize_polygon(polygon:Polygon, class_id:int, out:np.ndarray, all_touched=True):
    rasterize([(polygon, class_id)], out=out, all_touched=all_touched, default_value=0)

def extract_image_coordinates(classified_image:np.ndarray, class_id=1, ravel=True):
    assert classified_image.ndim == 2, "Expecting classified image to be two dimensions"
    # print(coordinates)
    # import ipdb; ipdb.set_trace()
    if ravel:
        coordinates = np.ma.where(classified_image == class_id)
        coordinates = np.ravel_multi_index(coordinates, classified_image.shape)
    else:
        coordinates = np.asarray(np.column_stack(np.ma.where(classified_image == class_id)))
    return coordinates
    