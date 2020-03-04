import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List

dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    img_dx = cv2.Scharr(img, cv2.CV_64FC1, 1, 0)
    img_dy = cv2.Scharr(img, cv2.CV_64FC1, 0, 1)

    img_dx2 = cv2.GaussianBlur(img_dx * img_dx, None, sigmaX=blur_sigma, sigmaY=blur_sigma)
    img_dy2 = cv2.GaussianBlur(img_dy * img_dy, None, sigmaX=blur_sigma, sigmaY=blur_sigma)
    img_dxdy = cv2.GaussianBlur(img_dx * img_dy, None, sigmaX=blur_sigma, sigmaY=blur_sigma)

    response_function = np.divide(img_dx2*img_dy2 - img_dxdy*img_dxdy, img_dx2 + img_dy2 + 1e-6)

    print(response_function.max())

    maxima = cv2.dilate(response_function, dilate_kernel, iterations=3)
    maxima[maxima < threshold] = 0.0

    points_image = np.logical_and(abs(maxima - response_function) < 1e-6, maxima >= threshold).astype(np.ubyte)

    points = cv2.findNonZero(points_image).squeeze()
    n_points, _ = points.shape

    img_vis = img.copy()  # We dont want to draw on the original image
    for i in range(n_points):
        u, v = points[i, :]
        cv2.circle(img_vis, (u, v), 3, 0, -1)
    cv2.imshow("img", img_vis)

    points_and_response = []

    for i in range(n_points):
        u, v = points[i, :]
        points_and_response.append((response_function[v, u], points[i, :]))

    points_and_response = sorted(points_and_response, key=itemgetter(0), reverse=True)

    return points_and_response
