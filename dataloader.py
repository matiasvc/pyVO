import cv2
import os
import numpy as np
from typing import Tuple
from pyquaternion import Quaternion


class DataLoader:

    def __init__(self, dataset_path: str):
        assert os.path.exists(dataset_path), f'Folder does not exist: {os.path.abspath(dataset_path)}'
        self.datasetPath = dataset_path

        self.grayImageIndex = 0
        rgb_file_path = os.path.join(dataset_path, 'rgb.txt')
        assert os.path.exists(rgb_file_path), f'Cant find rgb.txt file: {os.path.abspath(rgb_file_path)}'

        self.rgbImagePaths = []
        with open(rgb_file_path) as rgb_file:
            for line in rgb_file:
                if line[0] == '#':
                    continue
                timestamp_str, path_str = line.split(' ')
                timestamp = float(timestamp_str)
                self.rgbImagePaths.append((timestamp, path_str.strip()))

        self.depthImageIndex = 0
        depth_file_path = os.path.join(dataset_path, 'depth.txt')
        assert os.path.exists(depth_file_path), f'Cant find depth.txt file: {os.path.abspath(depth_file_path)}'

        self.depthImagePaths = []
        with open(depth_file_path) as depth_file:
            for line in depth_file:
                if line[0] == '#':
                    continue
                timestamp_str, path_str = line.split(' ')
                timestamp = float(timestamp_str)
                self.depthImagePaths.append((timestamp, path_str.strip()))

        self.gtIndex = 0
        gt_file_path = os.path.join(dataset_path, 'groundtruth.txt')
        assert os.path.exists(gt_file_path), f'Cant find groundtruth.txt file: {os.path.abspath(gt_file_path)}'

        self.gtTransforms = []
        with open(gt_file_path) as gt_file:
            for line in gt_file:
                if line[0] == '#':
                    continue
                timestamp_str, tx_str, ty_str, tz_str, qx_str, qy_str, qz_str, qw_str = line.split(' ')
                timestamp = float(timestamp_str)
                position = np.array([float(tx_str), float(ty_str), float(tz_str)])
                orientation = np.array([float(qx_str), float(qy_str), float(qz_str), float(qw_str)])
                self.gtTransforms.append((timestamp, position, orientation))

    def get_greyscale(self) -> np.ndarray:
        image_path = self.rgbImagePaths[self.grayImageIndex][1]
        image_path = os.path.join(self.datasetPath, image_path)
        img = cv2.imread(image_path, 0)  # Read and convert to 8-bit greyscale
        assert img is not None, f"Unable to read image: {os.path.abspath(image_path)}"
        return img.astype(np.float) / 255.0

    def get_depth(self) -> np.ndarray:
        current_timestamp = self.rgbImagePaths[self.grayImageIndex][0]
        # Find the depth image with timestamp closest to the timestamp of the current rgb image
        while self.depthImageIndex < len(self.depthImagePaths)-1 and abs(self.depthImagePaths[self.depthImageIndex+1][0] - current_timestamp) < abs(self.depthImagePaths[self.depthImageIndex][0] - current_timestamp):
            self.depthImageIndex += 1

        image_path = self.depthImagePaths[self.depthImageIndex][1]
        image_path = os.path.join(self.datasetPath, image_path)
        img = cv2.imread(image_path, -1)  # Read the image unchanged to get 16-bit images
        assert img is not None, f"Unable to read image: {os.path.abspath(image_path)}"
        return img.astype(np.int16)

    def get_transform(self) -> Tuple[Quaternion, np.ndarray]:
        current_timestamp = self.rgbImagePaths[self.grayImageIndex][0]
        # Find the transform with timestamp closest to the timestamp of the current rgb image
        while self.gtIndex < len(self.gtTransforms)-1 and abs(self.gtTransforms[self.gtIndex + 1][0] - current_timestamp) < abs(self.gtTransforms[self.gtIndex][0] - current_timestamp):
            self.gtIndex += 1

        _, t, orientation = self.gtTransforms[self.gtIndex]
        q = Quaternion(orientation)
        return q, t

    def next(self) -> None:
        self.grayImageIndex += 1

    def has_next(self) -> bool:
        return self.grayImageIndex < len(self.rgbImagePaths)
