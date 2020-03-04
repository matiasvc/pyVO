import numpy as np
import cv2

from pyflann import *
from typing import Tuple, List
from math import sin, cos, pi, sqrt
from collections import defaultdict

flann = FLANN()


def get_warped_patch(img: np.ndarray, patch_size: int,
                     x_translation: float, y_translation: float, theta) -> np.ndarray:
    """
    Returns a warped image patch.
    :param img: Original image.
    :param patch_size: The size of the patch. Should be a odd number.
    :param x_translation: The x position of the patch center.
    :param y_translation: The y position of the patch center.
    :param theta: The rotation of the patch in radians.
    :return: The warped image patch.
    """
    patch_half_size = patch_size/2
    c = cos(-theta)
    s = sin(-theta)

    t = np.array([[c, s, (-c - s) * patch_half_size + x_translation],
                  [-s, c, (s - c) * patch_half_size + y_translation]])

    return cv2.warpAffine(img, t, (patch_size, patch_size), flags=cv2.WARP_INVERSE_MAP)


class KLTTracker:

    def __init__(self, initial_position: np.ndarray, origin_image: np.ndarray, patch_size, tracker_id):
        assert patch_size >= 3 and patch_size % 2 == 1, f'patch_size must be 3 or greater and be a odd number, is {patch_size}'
        self.initialPosition = initial_position

        self.translationX = 0.0
        self.translationY = 0.0
        self.theta = 0.0

        self.positionHistory = [(self.pos_x, self.pos_y, self.theta)]
        self.trackerID = tracker_id
        self.patchSize = patch_size
        self.patchHalfSizeFloored = patch_size // 2

        pos_x, pos_y = initial_position
        image_height, image_width = origin_image.shape
        assert self.patchHalfSizeFloored <= pos_x < image_width - self.patchHalfSizeFloored \
               and self.patchHalfSizeFloored <= pos_y < image_height - self.patchHalfSizeFloored, \
            f'Point is to close to the image border for the current patch size, point is {initial_position} and patch_size is {patch_size}'
        self.trackingPatch = origin_image[pos_y - self.patchHalfSizeFloored:pos_y + self.patchHalfSizeFloored + 1,
                                          pos_x - self.patchHalfSizeFloored:pos_x + self.patchHalfSizeFloored + 1]
        self.visualizeColor = np.random.randint(0, 256, 3, dtype=int)
        self.patchBorder = sqrt(2*patch_size**2) + 1

    @property
    def pos_x(self):
        return self.initialPosition[0] + self.translationX

    @property
    def pos_y(self):
        return self.initialPosition[1] + self.translationY

    def track_new_image(self, img: np.ndarray, img_grad: np.ndarray, max_iterations: int,
                        min_delta_length=2.5e-2, max_error=0.035) -> int:
        """
        Tracks the KLT tracker on a new grayscale image. You will need the get_warped_patch function here.
        :param img: The image.
        :param img_grad: The image gradient.
        :param max_iterations: The maximum number of iterations to run.
        :param min_delta_length: The minimum length of the delta vector.
        If the length is shorter than this, then the optimization should stop.
        :param max_error: The maximum error allowed for a valid track.
        :return: Return 0 when track is successful, 1 any point of the tracking patch is outside the image,
        2 if a invertible hessian is encountered and 3 if the final error is larger than max_error.
        """

        for iteration in range(max_iterations):

            image_height, image_width = img.shape
            if self.pos_x - self.patchBorder < 0 or self.pos_x + self.patchBorder >= image_width or \
               self.pos_y - self.patchBorder < 0 or self.pos_y + self.patchBorder >= image_height:
                return 1

            image_patch = get_warped_patch(img, self.patchSize, self.pos_x, self.pos_y, self.theta)
            image_grad_patch = get_warped_patch(img_grad, self.patchSize, self.pos_x, self.pos_y, self.theta)

            patch_difference = self.trackingPatch - image_patch

            warp_jacobians = np.zeros((self.patchSize, self.patchSize, 2, 3))
            warp_jacobians[:, :, 0, 0] = 1.0
            warp_jacobians[:, :, 1, 1] = 1.0

            u_grid, v_grid = np.mgrid[-self.patchHalfSizeFloored: self.patchHalfSizeFloored + 1,
                                      -self.patchHalfSizeFloored: self.patchHalfSizeFloored + 1]

            s = sin(self.theta)
            c = cos(self.theta)
            warp_jacobians[:, :, 0, 2] = -s*u_grid - c*v_grid
            warp_jacobians[:, :, 1, 2] = c*u_grid - s*v_grid

            image_jacobians = image_grad_patch[:, :, np.newaxis] @ warp_jacobians
            steepest_descent = np.sum(image_jacobians * patch_difference[:, :, np.newaxis, np.newaxis], axis=(0, 1))
            hessian = np.sum(np.swapaxes(image_jacobians, 2, 3) @ image_jacobians, axis=(0, 1))

            try:
                summed_hessian_inv = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                return 2

            delta_p = summed_hessian_inv @ steepest_descent.T
            delta_x, delta_y, delta_theta = delta_p

            translation_step_length = 12
            rotation_step_length = 1

            self.translationX += translation_step_length*delta_x[0]
            self.translationY += translation_step_length*delta_y[0]
            self.theta += rotation_step_length*delta_theta[0]

            ### Uncomment these lines to visualize the patch optimization
            # cv2.imshow('tracking patch', self.trackingPatch)
            # cv2.imshow('image patch', image_patch)
            # cv2.imshow('patch difference', patch_difference)
            # cv2.waitKey(50)

            if np.linalg.norm(delta_p) < min_delta_length:
                break

        # Calculate final error
        image_patch = get_warped_patch(img, self.patchSize, self.pos_x, self.pos_y, self.theta)
        patch_difference = self.trackingPatch - image_patch
        avg_error = np.sum(np.square(patch_difference)) / self.patchSize**2

        self.positionHistory.append((self.pos_x, self.pos_y, self.theta))

        if avg_error >= max_error:
            return 3
        else:
            return 0


class PointTracker:

    def __init__(self, max_points=80, tracking_patch_size=27):
        self.maxPoints = max_points
        self.trackingPatchSize = tracking_patch_size
        self.currentTrackers = []
        self.nextTrackerId = 0

    def visualize(self, img: np.ndarray, draw_id=False):
        img_vis = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for klt in self.currentTrackers:
            x_pos = int(round(klt.pos_x))
            y_pos = int(round(klt.pos_y))
            length = 20
            x2_pos = int(round(x_pos + length * cos(-klt.theta + pi / 2)))
            y2_pos = int(round(y_pos - length * sin(-klt.theta + pi / 2)))
            cv2.circle(img_vis, (x_pos, y_pos), 3, [int(c) for c in klt.visualizeColor], -1)
            cv2.line(img_vis, (x_pos, y_pos), (x2_pos, y2_pos), 0, thickness=1, lineType=cv2.LINE_AA)
            if draw_id:
                cv2.putText(img_vis, f'{klt.trackerID}', (x_pos+5, y_pos), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 100, 0))

            if len(klt.positionHistory) >= 2:
                for i in range(len(klt.positionHistory)-1):
                    x_from, y_from, _ = klt.positionHistory[i]
                    x_to, y_to, _ = klt.positionHistory[i+1]
                    cv2.line(img_vis, (int(round(x_from)), int(round(y_from))), (int(round(x_to)), int(round(y_to))), 0, thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow("KLT Trackers", img_vis)

    def add_new_corners(self, origin_image: np.ndarray, points_and_response_list: List[Tuple[float, np.ndarray]],
                        min_distance=13.0) -> None:
        assert len(points_and_response_list) > 0, 'points_list is empty'

        for i in range(len(points_and_response_list) - 1):  # Check that points_list is sorted from largest to smallest response value
            assert points_and_response_list[i][0] >= points_and_response_list[i + 1][0], 'points_list is not sorted'

        if len(self.currentTrackers) >= self.maxPoints:  # Dont do anything if we already have the maximum number of points
            return

        filtered_points = []

        image_height, image_width = origin_image.shape
        patch_border = sqrt(2 * self.trackingPatchSize ** 2) + 1
        for _, point in points_and_response_list:  # Filter out points to close to the image border
            pos_x, pos_y = point
            if patch_border <= pos_x < image_width - patch_border \
                    and patch_border <= pos_y < image_height - patch_border:
                filtered_points.append(point)

        points = filtered_points
        filtered_points = []
        if len(self.currentTrackers) > 0:  # Filter out points to close to existing points
            current_points = [np.array([klt.pos_x, klt.pos_y]) for klt in self.currentTrackers]
            _, dists = flann.nn(np.array(current_points, dtype=np.int32), np.array(points, dtype=np.int32), 1)
            dists = np.sqrt(dists)
            filter_indices = np.arange(0, len(points))[dists >= min_distance]

            for i in filter_indices:
                filtered_points.append(points[i])
            points = filtered_points

        # Add at most enough points to bring us up to the max number of points
        number_of_points_to_add = min(len(points), self.maxPoints - len(self.currentTrackers))
        points = points[:number_of_points_to_add]

        for point in points:
            self.currentTrackers.append(KLTTracker(point, origin_image, self.trackingPatchSize, self.nextTrackerId))
            self.nextTrackerId += 1

    def track_on_image(self, img: np.ndarray, max_iterations=25) -> None:

        img_dx = cv2.Scharr(img, cv2.CV_64FC1, 1, 0)
        img_dy = cv2.Scharr(img, cv2.CV_64FC1, 0, 1)
        img_grad = np.stack((img_dx, img_dy), axis=-1)

        lost_track = []
        tracker_return_values = defaultdict(int)
        for klt in self.currentTrackers:
            tracker_condition = klt.track_new_image(img, img_grad, max_iterations)
            tracker_return_values[tracker_condition] += 1
            if tracker_condition != 0:
                lost_track.append(klt)

        print(f"Tracked frame - remained: {tracker_return_values[0]}, hit border: {tracker_return_values[1]}, "
              f"singular_hessian: {tracker_return_values[2]}, large error: {tracker_return_values[3]}")

        for klt in lost_track:
            self.currentTrackers.remove(klt)

    def get_position_with_id(self) -> Tuple[np.ndarray, np.ndarray]:
        n_points = len(self.currentTrackers)
        ids = np.empty(n_points, dtype=np.int32)
        positions = np.empty((2, n_points), dtype=np.float64)

        for i, klt in enumerate(self.currentTrackers):
            ids[i] = klt.trackerID
            positions[:, i] = (klt.pos_x, klt.pos_y)

        return ids, positions
