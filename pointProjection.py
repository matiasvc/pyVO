import numpy as np
import cv2

from typing import Tuple

filter_size = 13

assert filter_size % 2 == 1, 'Filter size must be a odd number'
filter_half_size = filter_size // 2
filter_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size)) == 1

fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7


def project_points(ids: np.ndarray, points: np.ndarray, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the 2D points to 3D using the depth image and the camera instrinsic parameters.
    :param ids: A N vector point ids.
    :param points: A 2xN matrix of 2D points
    :param depth_img: The depth image. Divide pixel value by 5000 to get depth in meters.
    :return: A tuple containing a N vector and a 3xN vector of all the points that where successfully projected.
    """
    cv2.imshow('depth', depth_img/np.max(depth_img))

    n_points = len(ids)

    image_height, image_width = depth_img.shape

    projected_points = []
    new_ids = []

    for i in range(n_points):
        pos_x, pos_y = points[:, i]
        pos_xi = int(round(pos_x))
        pos_yi = int(round(pos_y))

        if pos_xi < filter_half_size or pos_xi >= image_width - filter_half_size or \
           pos_yi < filter_half_size or pos_yi >= image_height - filter_half_size:
            continue

        depth_patch = depth_img[pos_yi-filter_half_size:pos_yi+filter_half_size+1, pos_xi-filter_half_size:pos_xi+filter_half_size+1]
        depth_patch = depth_patch[np.logical_and(filter_kernel,  depth_patch != 0)]

        if len(depth_patch) == 0:
            continue

        z = np.min(depth_patch) / 5000.0

        u, v = points[:, i]

        x = (u - cx)*z / fx
        y = (v - cy)*z / fy

        projected_points.append((x, y, z))
        new_ids.append(ids[i])

    return np.array(new_ids), np.array(projected_points).T

