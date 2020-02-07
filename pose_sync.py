#!/usr/bin/env python3

import argparse
import sys

import mpl_toolkits.mplot3d.proj3d as proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import expm, logm
from math import sin, cos
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import Lie

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sigma-scale'
    )
    parser.add_argument(
        '--max-iter'
    )
    parser.add_argument(
        '--no-noise',
        help='Odom observation is perfect.',
        action='store_true')

    args = parser.parse_args()

    np.random.seed(0)
    U = [Lie.from_rpy_xyz(0., 0., 0., 2, 1, 0),  # T1 + T01 = T0
         Lie.from_rpy_xyz(0., 0., 0., 2, 1, 0),
         Lie.from_rpy_xyz(0., 0., 0., 0, 2, 1),
         Lie.from_rpy_xyz(0., 0., 0., -2, -1, 0),
         Lie.from_rpy_xyz(0., 0., 0., -2, -1, 0),
    ]

    poses = [np.eye(4)]
    pose_links = []
    for i, u in enumerate(U):
        poses.append(poses[-1].dot(u))
        pose_links.append((i, i + 1))

    kStdScale = float(args.sigma_scale) if args.sigma_scale else 1
    Sigma = np.diag([0.05**2, 0.03**2, 0.03**2, 0.05**2, 0.05**2, 0.05**2])
    L = np.linalg.cholesky(Sigma * kStdScale**2)
    odom_Z = []
    for u in U:
        noise = L.dot(np.random.normal(0, 1, (6, 1)))
        odom_Z.append(u.dot(expm(Lie.hat(noise))))

    loop_Z = []
    # Add a loop closure a, b
    for i in range(0, 1):
        a = 0 if i == 0 else np.random.randint(0, len(poses) - 1)
        b = len(poses) - 1
        pose_links.append((a, b))
        # noise = L.dot(np.random.normal(0, 1, (6, 1)))
        # Z = Tab = Ta.inv() * Tb: Ideal measurement
        loop_Z.append(np.linalg.inv(poses[a]).dot(poses[b]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plot_poses(ax, poses, pose_links, 'r', True)

    poses_init = [np.eye(4)]
    for z in odom_Z:
        poses_init.append(poses_init[-1].dot(z))

    plot_poses(ax, poses_init, pose_links, 'g')

    # We now construct the Jacobian matrix. The rows are the measurements which are
    # SE(3) here. Therefore, each measurement occupies 6 rows. Columns
    # correspond to the decision variables which are SE(3) here. Therefore, we
    # have 6 x number of poses variables. Note that the optimization is
    # parametrized uing twist (6x1) which lives in the Lie algebra se(3). We
    # find the correction twist and retract it using Lie exp map and "add"
    # (multiply) it to the previous iteration guess. This process should be
    # repeated for each pose before moving to the next iteration. Further, we
    # anchor the first pose to constrain the graph, i.e., we fix the first pose
    # to be at the identity. This will add an extra 6 rows to the jacobian
    # matrix and is equivalent of placing a prior over the first node of the
    # graph.

    # Jacobian
    Z = U + loop_Z if args.no_noise else odom_Z + loop_Z
    J = np.zeros((6 + 6 * len(Z), 6 * len(poses_init)))
    r = np.zeros((6 + 6 * len(Z), 1))
    last_r = np.zeros((6 + 6 * len(Z), 1))
    cov = np.eye(6 + 6 * len(Z))
    # Set supper small variance for first measurement
    cov[0:6, 0:6] *= 0.000001
    # Trust more of loop closures
    cov[6*len(odom_Z):6+6*len(Z), 6*len(odom_Z):6+6*len(Z)] *= 0.01
    max_iter = int(args.max_iter) if args.max_iter else 50
    iter = 0
    r_eps = 1e-9
    poses_est = poses_init
    while iter < max_iter:
        print("Iter: ", iter)
        iter += 1
        r[0:6] = Lie.vee(logm(poses_est[0]))
        J[0:6, 0:6] = np.eye(6)
        for i, (a, b) in enumerate(pose_links):
            # Measurement is T_ab = T_a.inv() * T_b
            # ||J*dx + h(x) - z||_2
            # r = h(x) - z = vee(log(T_a.inv()*T_b)) - vee(log(T_ab))
            # r = vee(log(T_a.inv()*T_b)) - vee(log(T_ab))
            # r = vee(log(T_ba * (T_a.inv()*T_b)))
            res_idx = 6*(i+1)
            res = Lie.vee(np.linalg.inv(Z[i]).dot(
                np.linalg.inv(poses_est[a]).dot(poses_est[b])))
            r[res_idx:res_idx+6] = res
            # Measurement is T_ab = T_a.inv() * T_b
            T_ba = np.linalg.inv(poses_est[b]).dot(poses_est[a])
            J[res_idx:res_idx+6, a*6:(a+1)*6] = - \
                Lie.RightJacobianInverse_SE3(res).dot(Lie.Adjoint_SE3(T_ba))
            J[res_idx:res_idx+6, b *
                6:(b+1)*6] = Lie.RightJacobianInverse_SE3(res)
        # solve normal equations
        A = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), J])
        b = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), -r])
        dx = np.linalg.inv(A).dot(b)
        for i in range(len(poses_est)):
            poses_est[i] = poses_est[i].dot(expm(Lie.hat(dx[i*6:(i+1)*6])))
        print("residual: ", np.linalg.norm(r))
        if np.linalg.norm(r - last_r) < r_eps:
            break
        last_r[:] = r[:]
    plot_poses(ax, poses_est, pose_links, 'b')
    print("Pose to pose links: \n", pose_links)
    plt.show()


def plot_poses(ax, poses, pose_links, c, show_links=False):
    ax.set_xlim3d(0, 5)
    ax.set_ylim3d(0, 5)
    ax.set_zlim3d(0, 5)
    kAxisLen = 1
    for T in poses:
        x = np.matmul(T, np.array([kAxisLen, 0, 0, 1]))
        y = np.matmul(T, np.array([0, kAxisLen, 0, 1]))
        z = np.matmul(T, np.array([0, 0, kAxisLen, 1]))
        x_axis = Arrow3D([T[0, 3], x[0]], [T[1, 3], x[1]], [T[2, 3], x[2]], mutation_scale=20,
                         lw=2, arrowstyle="-|>", color=c)
        y_axis = Arrow3D([T[0, 3], y[0]], [T[1, 3], y[1]], [T[2, 3], y[2]], mutation_scale=20,
                         lw=2, arrowstyle="-|>", color=c)
        z_axis = Arrow3D([T[0, 3], z[0]], [T[1, 3], z[1]], [T[2, 3], z[2]], mutation_scale=20,
                         lw=2, arrowstyle="-|>", color=c)
        ax.add_artist(x_axis)
        ax.add_artist(y_axis)
        ax.add_artist(z_axis)
    if not show_links:
        return
    for (i, j) in pose_links:
        ta = [poses[i][0, 3], poses[i][1, 3], poses[i][2, 3]]
        tb = [poses[j][0, 3], poses[j][1, 3], poses[j][2, 3]]
        if j-i == 1:
            ta_tb = Arrow3D([ta[0], tb[0]], [ta[1], tb[1]], [ta[2], tb[2]], mutation_scale=20,
                            lw=0.5, arrowstyle="-|>", color="k")
        else:
            ta_tb = Arrow3D([ta[0], tb[0]], [ta[1], tb[1]], [ta[2], tb[2]], mutation_scale=20,
                            lw=2, arrowstyle="-|>", color="y")
        ax.add_artist(ta_tb)


if __name__ == '__main__':
    sys.exit(main())
