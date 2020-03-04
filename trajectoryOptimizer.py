import numpy as np
from liegroups import SE3
from pyquaternion import Quaternion
from typing import Tuple

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

global frame
frame = 0

def plot_points(points, transformedPoints):
    global frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points, c='r')
    ax.scatter(*transformedPoints, c='b')

    _, n_points = points.shape

    for i in range(n_points):
        pointA = points[:,i]
        pointB = transformedPoints[:,i]
        xs = [pointA[0], pointB[0]]
        ys = [pointA[1], pointB[1]]
        zs = [pointA[2], pointB[2]]
        ax.plot(xs, ys, zs)

    #plt.savefig(f'img/frame{frame:03d}.png')
    plt.show()
    frame += 1

def optimize_transformation(previous_ids: np.ndarray, previous_points: np.ndarray,
                            current_ids: np.ndarray, current_points: np.ndarray) -> Tuple[Quaternion, np.ndarray]:
    common_points = set(previous_ids).intersection(set(current_ids))

    if len(common_points) < 3:
        print('WARNING: Fewer than 3 common points. Unable to find transfomration')
        return Quaternion(), np.zeros(3)

    previous_points_filter = np.array([i for i in range(len(previous_ids)) if previous_ids[i] in common_points])
    current_points_filter = np.array([i for i in range(len(current_ids)) if current_ids[i] in common_points])

    previous_points = previous_points[:, previous_points_filter]
    current_points = current_points[:, current_points_filter]

    _, n_points = previous_points.shape

    xi = np.zeros(6)

    plot_points(previous_points, current_points)

    weights = np.ones(len(common_points))

    for attempt in range(5):

        for iteration in range(10):
            A = np.zeros((3*n_points, 6))
            b = np.zeros(3*n_points)

            T = SE3.exp(xi)
            R = T.rot.as_matrix()
            t = T.trans

            iteration_points = R @ previous_points + t[:, np.newaxis]
            errors = np.sqrt(np.sum(np.square(current_points - iteration_points), axis=0))

            k = 1.345 * np.std(errors)
            huber_weights = np.array(list(map(lambda x: 1.0 if x <= k else k/abs(x), errors))) * weights
            huber_weights = np.repeat(huber_weights, 3)

            W = np.diag(huber_weights)

            for i in range(n_points):
                x1, x2, x3 = iteration_points[:, i]
                A[i * 3:i * 3 + 3, :] = np.array([[1, 0, 0,   0,  x3, -x2],
                                                  [0, 1, 0, -x3,   0,  x1],
                                                  [0, 0, 1,  x2, -x1,   0]])

                b[i * 3:i * 3 + 3] = current_points[:, i] - iteration_points[:, i]

            AA = A.T @ W @ A
            AA_inv = np.linalg.inv(AA)
            xi_delta = AA_inv @ A.T @ W @ b
            xi = SE3.log(SE3.exp(xi)*SE3.exp(xi_delta))

            if np.linalg.norm(xi_delta) < 1e-8:
                break

        for i in range(len(common_points)):
            weights[i] = 0 if errors[i] > np.average(errors) + 1*np.std(errors) else 1

    print(weights)
    T = SE3.exp(xi)
    R = T.rot.as_matrix()
    t = T.trans
    print(f'error: {np.max(errors)}, {np.min(errors)}, {np.average(errors)}, {np.std(errors)}')

    plot_points(iteration_points, current_points)
    return Quaternion(matrix=R), t
