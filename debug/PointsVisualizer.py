from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl
import pyqtgraph.Transform3D as Transform3D
from debug.GLRGBAxisItem import GLRGBAxisItem
from debug.GLFrustumItem import GLFrustumItem
from pyquaternion import Quaternion
import numpy as np


class PointVisualizer:

    def __init__(self):

        self.app = QtGui.QApplication([])
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 10
        self.view.show()
        self.view.setWindowTitle("Points Visualizer")

        grid = gl.GLGridItem(size=QtGui.QVector3D(10, 10, 1))
        grid.setSpacing(0.5, 0.5, 0.5)
        self.view.addItem(grid)

        axis = GLRGBAxisItem(size=QtGui.QVector3D(0.5, 0.5, 0.5))
        self.view.addItem(axis)

        self.gtFrustum = GLFrustumItem(frustumColor=(0.8, 0, 0, 0.6), size=QtGui.QVector3D(0.5, 0.5, 0.5))
        self.view.addItem(self.gtFrustum)

        self.gtLine = gl.GLLinePlotItem(color=(0.8, 0, 0, 0.6))
        self.view.addItem(self.gtLine)
        self.gtPositions = []

        self.estimatedFrustum = GLFrustumItem(frustumColor=(1, 1, 1, 0.6), size=QtGui.QVector3D(0.5, 0.5, 0.5))
        self.view.addItem(self.estimatedFrustum)

        self.estimatedLine = gl.GLLinePlotItem(color=(1, 1, 1, 0.6))
        self.view.addItem(self.estimatedLine)
        self.estimatedPositions = []

        self.points = gl.GLScatterPlotItem(color=(0, 0.8, 0, 1), size=3.0)
        self.points.setData(pos=np.zeros((1, 3)))
        self.view.addItem(self.points)

    def set_groundtruth_transform(self, orientation: Quaternion, position: np.ndarray):
        transform = Transform3D()
        transform.translate(*position)

        self.gtPositions.append(position)
        self.gtLine.setData(pos=np.array(self.gtPositions))

        qx, qy, qz, qw = orientation
        q = QtGui.QQuaternion(qw, qx, qy, qz)

        transform.rotate(q)
        self.gtFrustum.setTransform(transform)

    def set_estimated_transform(self, orientation: Quaternion, position: np.ndarray):
        transform = Transform3D()
        transform.translate(*position)

        self.estimatedPositions.append(position)
        self.estimatedLine.setData(pos=np.array(self.estimatedPositions))

        qx, qy, qz, qw = orientation
        q = QtGui.QQuaternion(qw, qx, qy, qz)

        transform.rotate(q)
        self.estimatedFrustum.setTransform(transform)

    def set_projected_points(self, points: np.ndarray, orientation: Quaternion, position: np.ndarray):

        if len(points) == 0:
            return

        R = orientation.rotation_matrix
        points = R @ points + position[:, np.newaxis]

        self.points.setData(pos=points.T)
