#!/usr/bin/env python3

import Lie
from g2o_reader import read_g2o
import argparse
import sys
import copy

import mpl_toolkits.mplot3d.proj3d as proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import expm, logm
from math import sin, cos, sqrt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=["sync", "matrix"], required=True)
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
    parser.add_argument(
        '--g2o-file',
    )
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
    L = cholesky(Sigma * kStdScale**2)
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

    Z = U + loop_Z if args.no_noise else odom_Z + loop_Z

    if args.g2o_file is not None:
        (poses_init, Z, pose_links) = read_g2o(args.g2o_file)
        poses = poses_init
    # ax.clear()
    # print(poses_init[0])
    # plot_poses(ax, poses_init, pose_links, 'g')
    # plt.show()
    # return

    # Jacobian
    J, r, last_r, cov = init_for_sync(
        poses_init=poses_init, odom_Z=odom_Z, Z=Z) if args.mode == "sync" else init_for_matrix(
        poses_init=poses_init, odom_Z=odom_Z, Z=Z)
    max_iter = int(args.max_iter) if args.max_iter else 50
    iter = 0
    r_eps = 1e-2
    poses_est = copy.deepcopy(poses_init)
    while iter < max_iter:
        print("Iter: ", iter)
        iter += 1
        if args.mode == "sync":
            update_J_r_for_sync(poses_init=poses_init, poses_est=poses_est,
                                pose_links=pose_links, Z=Z, cov=cov, J=J, r=r)
        else:
            update_J_r_for_matrix(poses_init=poses_init, poses_est=poses_est,
                                  pose_links=pose_links, Z=Z, cov=cov, J=J, r=r)

        if np.linalg.norm(r - last_r) < r_eps:
            break
        last_r[:] = r[:]
        ax.clear()
        plot_poses(ax, poses, pose_links, 'r', True)
        plot_poses(ax, poses_init, pose_links, 'g')
        plot_poses(ax, poses_est, pose_links, 'b')
        plt.pause(0.2)

    ax.clear()
    plot_poses(ax, poses, pose_links, 'r', True)
    plot_poses(ax, poses_init, pose_links, 'g')
    plot_poses(ax, poses_est, pose_links, 'b')
    print("Pose to pose links: \n", pose_links)
    plt.show()


def init_for_sync(*, poses_init, odom_Z, Z):
    J = np.zeros((6 + 6 * len(Z), 6 * len(poses_init)))
    r = np.zeros((6 + 6 * len(Z), 1))
    last_r = np.zeros((6 + 6 * len(Z), 1))
    cov = np.eye(6 + 6 * len(Z))
    # Set supper small variance for first measurement
    cov[0:6, 0:6] *= 0.000001
    # Trust more of loop closures
    cov[6 * len(odom_Z):6 + 6 * len(Z), 6 * len(odom_Z):6 + 6 * len(Z)] *= 0.01
    return (J, r, last_r, cov)


def update_J_r_for_sync(*, poses_init, poses_est, pose_links, Z, cov, J, r):
    r[0:6] = Lie.vee(logm(np.linalg.inv(poses_init[0]).dot(poses_est[0])))
    J[0:6, 0:6] = Lie.RightJacobianInverse_SE3(r[0:6])
    for i, (a, b) in enumerate(pose_links):
        # Measurement is T_ab = T_a.inv() * T_b
        # ||J*dx + h(x) - z||_2
        # r = h(x) - z = vee(log(T_a.inv()*T_b)) - vee(log(T_ab))
        # r = vee(log(T_a.inv()*T_b)) - vee(log(T_ab))
        # r = vee(log(T_ba * (T_a.inv()*T_b)))
        res_idx = 6 * (i + 1)
        res = Lie.vee(np.linalg.inv(Z[i]).dot(
            np.linalg.inv(poses_est[a]).dot(poses_est[b])))
        r[res_idx:res_idx + 6] = res
        # Measurement is T_ab = T_a.inv() * T_b
        T_ba = np.linalg.inv(poses_est[b]).dot(poses_est[a])
        J[res_idx:res_idx + 6, a * 6:(a + 1) * 6] = - \
            Lie.RightJacobianInverse_SE3(res).dot(Lie.Adjoint_SE3(T_ba))
        J[res_idx:res_idx + 6, b *
          6:(b + 1) * 6] = Lie.RightJacobianInverse_SE3(res)
        # solve normal equations
    A = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), J])
    b = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), -r])
    dx = solve_Ab(A,b)
    # dx = np.linalg.inv(A).dot(b)
    for i in range(len(poses_est)):
        poses_est[i] = poses_est[i].dot(expm(Lie.hat(dx[i * 6:(i + 1) * 6])))
    print("residual: ", np.linalg.norm(r))


def init_for_matrix(*, poses_init, odom_Z, Z):
    J = np.zeros((12 + 12 * len(Z), 12 * len(poses_init)))
    r = np.zeros((12 + 12 * len(Z), 1))
    last_r = np.zeros((12 + 12 * len(Z), 1))
    cov = np.eye(12 + 12 * len(Z))
    # Set supper small variance for first measurement
    cov[0:12, 0:12] *= 0.000001
    # Trust more of loop closures
    cov[12 * len(odom_Z):12 + 12 * len(Z), 12 * len(odom_Z):12 + 12 * len(Z)] *= 0.01
    return (J, r, last_r, cov)


def update_J_r_for_matrix(*, poses_init, poses_est, pose_links, Z, cov, J, r):
    r[0:9] = (poses_est[0][0:3, 0:3] - np.eye(3)).reshape(9, 1)
    r[9:12] = poses_est[0][0:3, [3]] - poses_init[0][0:3, [3]]
    J[:12, :12] = np.eye(12)
    for i, (a, b) in enumerate(pose_links):
        res_idx = 12 * (i + 1)
        T_ba_z = np.linalg.inv(Z[i])
        T_a = poses_est[a]
        T_b = poses_est[b]
        R_a = np.zeros((4, 4))
        R_a[1:4, 1:4] = T_a[:3, :3]
        R_b = np.zeros((4, 4))
        R_b[1:4, 1:4] = T_b[:3, :3]
        t_a = T_a[0:3, [3]]
        t_b = T_b[0:3, [3]]
        T_ab = np.linalg.inv(T_a).dot(T_b)
        r[res_idx:res_idx +
          9] = (T_ba_z[:3, :3].dot(T_ab[:3, :3]) - np.eye(3)).reshape(9, 1)
        r[res_idx + 9:res_idx + 12] = T_ba_z[:3,
                                             :3].dot(T_ab[:3, [3]]) + T_ba_z[:3, [3]]
        base_a = a * 12
        tb_minus_ta = t_b - t_a
        J[res_idx, base_a:base_a + 9] = np.array([[R_b[1, 1] * R_a[1, 1], R_b[1, 1] * R_a[1, 2], R_b[1, 1] * R_a[1, 3],
                                                   R_b[2, 1] * R_a[1, 1], R_b[2, 1] *
                                                   R_a[1, 2], R_b[2, 1] *
                                                   R_a[1, 3],
                                                   R_b[3, 1] * R_a[1, 1], R_b[3, 1] * R_a[1, 2], R_b[3, 1] * R_a[1, 3]]])
        J[res_idx + 1, base_a:base_a + 9] = np.array([[R_b[1, 2] * R_a[1, 1], R_b[1, 2] * R_a[1, 2], R_b[1, 2] * R_a[1, 3],
                                                       R_b[2, 2] * R_a[1, 1], R_b[2, 2] *
                                                       R_a[1, 2], R_b[2,
                                                                      2] * R_a[1, 3],
                                                       R_b[3, 2] * R_a[1, 1], R_b[3, 2] * R_a[1, 2], R_b[3, 3] * R_a[1, 3]]])
        J[res_idx + 2, base_a:base_a + 9] = np.array([[R_b[1, 3] * R_a[1, 1], R_b[1, 3] * R_a[1, 3], R_b[1, 3] * R_a[1, 3],
                                                       R_b[2, 3] * R_a[1, 1], R_b[2, 3] *
                                                       R_a[1, 3], R_b[2,
                                                                      3] * R_a[1, 3],
                                                       R_b[3, 3] * R_a[1, 1], R_b[3, 3] * R_a[1, 3], R_b[3, 3] * R_a[1, 3]]])

        J[res_idx + 3, base_a:base_a + 9] = np.array([[R_b[1, 1] * R_a[2, 1], R_b[1, 1] * R_a[2, 2], R_b[1, 1] * R_a[2, 3],
                                                       R_b[2, 1] * R_a[2, 1], R_b[2, 1] *
                                                       R_a[2, 2], R_b[2,
                                                                      1] * R_a[2, 3],
                                                       R_b[3, 1] * R_a[2, 1], R_b[3, 1] * R_a[2, 2], R_b[3, 1] * R_a[2, 3]]])
        J[res_idx + 4, base_a:base_a + 9] = np.array([[R_b[1, 2] * R_a[2, 1], R_b[1, 2] * R_a[2, 2], R_b[1, 2] * R_a[2, 3],
                                                       R_b[2, 2] * R_a[2, 1], R_b[2, 2] *
                                                       R_a[2, 2], R_b[2,
                                                                      2] * R_a[2, 3],
                                                       R_b[3, 2] * R_a[2, 1], R_b[3, 2] * R_a[2, 2], R_b[3, 3] * R_a[2, 3]]])
        J[res_idx + 5, base_a:base_a + 9] = np.array([[R_b[1, 3] * R_a[2, 1], R_b[1, 3] * R_a[2, 3], R_b[1, 3] * R_a[2, 3],
                                                       R_b[2, 3] * R_a[2, 1], R_b[2, 3] *
                                                       R_a[2, 3], R_b[2,
                                                                      3] * R_a[2, 3],
                                                       R_b[3, 3] * R_a[2, 1], R_b[3, 3] * R_a[2, 3], R_b[3, 3] * R_a[2, 3]]])

        J[res_idx + 6, base_a:base_a + 9] = np.array([[R_b[1, 1] * R_a[3, 1], R_b[1, 1] * R_a[3, 2], R_b[1, 1] * R_a[3, 3],
                                                       R_b[2, 1] * R_a[3, 1], R_b[2, 1] *
                                                       R_a[3, 2], R_b[2,
                                                                      1] * R_a[3, 3],
                                                       R_b[3, 1] * R_a[3, 1], R_b[3, 1] * R_a[3, 2], R_b[3, 1] * R_a[3, 3]]])
        J[res_idx + 7, base_a:base_a + 9] = np.array([[R_b[1, 2] * R_a[3, 1], R_b[1, 2] * R_a[3, 2], R_b[1, 2] * R_a[3, 3],
                                                       R_b[2, 2] * R_a[3, 1], R_b[2, 2] *
                                                       R_a[3, 2], R_b[2,
                                                                      2] * R_a[3, 3],
                                                       R_b[3, 2] * R_a[3, 1], R_b[3, 2] * R_a[3, 2], R_b[3, 3] * R_a[3, 3]]])
        J[res_idx + 8, base_a:base_a + 9] = np.array([[R_b[1, 3] * R_a[3, 1], R_b[1, 3] * R_a[3, 3], R_b[1, 3] * R_a[3, 3],
                                                       R_b[2, 3] * R_a[3, 1], R_b[2, 3] *
                                                       R_a[3, 3], R_b[2,
                                                                      3] * R_a[3, 3],
                                                       R_b[3, 3] * R_a[3, 1], R_b[3, 3] * R_a[3, 3], R_b[3, 3] * R_a[3, 3]]])

        J[res_idx + 9, base_a] = tb_minus_ta[0]
        J[res_idx + 9, base_a + 3] = tb_minus_ta[0]
        J[res_idx + 9, base_a + 6] = tb_minus_ta[0]
        J[res_idx + 10, base_a + 1] = tb_minus_ta[1]
        J[res_idx + 10, base_a + 4] = tb_minus_ta[1]
        J[res_idx + 10, base_a + 7] = tb_minus_ta[1]
        J[res_idx + 11, base_a + 2] = tb_minus_ta[2]
        J[res_idx + 11, base_a + 5] = tb_minus_ta[2]
        J[res_idx + 11, base_a + 8] = tb_minus_ta[2]

        J[res_idx + 9:res_idx + 12, base_a + 0:base_a +
          3] = np.array([[tb_minus_ta[0], tb_minus_ta[0], tb_minus_ta[0]]])
        J[res_idx + 9:res_idx + 12, base_a + 3:base_a +
          6] = np.array([[tb_minus_ta[1], tb_minus_ta[1], tb_minus_ta[1]]])
        J[res_idx + 9:res_idx + 12, base_a + 6:base_a +
          9] = np.array([[tb_minus_ta[2], tb_minus_ta[2], tb_minus_ta[2]]])

        J[res_idx + 9:res_idx + 12, base_a + 9:base_a + 12] = - \
            T_ba_z[:3, :3].dot(T_a[:3, :3].transpose())
        #
        base_b = b * 12
        R_ji_ig = T_ba_z[:3, :3].dot(T_a[:3, :3].transpose())
        J[res_idx, base_b] = R_ji_ig[0, 0]
        J[res_idx, base_b + 3] = R_ji_ig[0, 1]
        J[res_idx, base_b + 6] = R_ji_ig[0, 2]
        J[res_idx + 1, base_b + 1] = R_ji_ig[0, 0]
        J[res_idx + 1, base_b + 4] = R_ji_ig[0, 1]
        J[res_idx + 1, base_b + 7] = R_ji_ig[0, 2]
        J[res_idx + 2, base_b + 2] = R_ji_ig[0, 0]
        J[res_idx + 2, base_b + 5] = R_ji_ig[0, 1]
        J[res_idx + 2, base_b + 8] = R_ji_ig[0, 2]

        J[res_idx + 3, base_b] = R_ji_ig[1, 0]
        J[res_idx + 3, base_b + 3] = R_ji_ig[1, 1]
        J[res_idx + 3, base_b + 6] = R_ji_ig[1, 2]
        J[res_idx + 4, base_b + 1] = R_ji_ig[1, 0]
        J[res_idx + 4, base_b + 4] = R_ji_ig[1, 1]
        J[res_idx + 4, base_b + 7] = R_ji_ig[1, 2]
        J[res_idx + 5, base_b + 2] = R_ji_ig[1, 0]
        J[res_idx + 5, base_b + 5] = R_ji_ig[1, 1]
        J[res_idx + 5, base_b + 8] = R_ji_ig[1, 2]

        J[res_idx + 6, base_b] = R_ji_ig[2, 0]
        J[res_idx + 6, base_b + 3] = R_ji_ig[2, 1]
        J[res_idx + 6, base_b + 6] = R_ji_ig[2, 2]
        J[res_idx + 7, base_b + 1] = R_ji_ig[2, 0]
        J[res_idx + 7, base_b + 4] = R_ji_ig[2, 1]
        J[res_idx + 7, base_b + 7] = R_ji_ig[2, 2]
        J[res_idx + 8, base_b + 2] = R_ji_ig[2, 0]
        J[res_idx + 8, base_b + 5] = R_ji_ig[2, 1]
        J[res_idx + 8, base_b + 8] = R_ji_ig[2, 2]

        J[res_idx + 9:res_idx + 12, base_b + 9:base_b +
            12] = T_ba_z[:3, :3].dot(T_a[:3, :3].transpose())
    A = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), J])
    b = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), -r])
    dx = np.linalg.inv(A).dot(b)
    for i in range(len(poses_est)):
        poses_est[i][0:3, 0:3] = poses_est[i][0:3, 0:3] + \
            dx[12 * i:12 * i + 9].reshape(3, 3)
        # Polarization
        U, _, V = np.linalg.svd(poses_est[i][0:3, 0:3], full_matrices=True)
        S = np.eye(3)
        S[2, 2] = 1 if np.linalg.det(U) * np.linalg.det(V) > 0 else -1
        poses_est[i][0:3, 0:3] = U.dot(S).dot(V)
        # print(dx[12*i+9:12*i+12])
        poses_est[i][0:3, [3]] = poses_est[i][0:3, [3]] + dx[12 * i + 9:12 * i + 12]
    print("residual: ", np.linalg.norm(r))


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


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
        if j - i == 1:
            ta_tb = Arrow3D([ta[0], tb[0]], [ta[1], tb[1]], [ta[2], tb[2]], mutation_scale=20,
                            lw=0.5, arrowstyle="-|>", color="k")
        else:
            ta_tb = Arrow3D([ta[0], tb[0]], [ta[1], tb[1]], [ta[2], tb[2]], mutation_scale=20,
                            lw=2, arrowstyle="-|>", color="y")
        ax.add_artist(ta_tb)


def cholesky(M):
    assert M.shape[0] == M.shape[1]
    N = M.shape[0]
    L = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            temp = 0
            if (j == i):
                for k in range(j):
                    temp += pow(L[j, k], 2)
                L[j, j] = sqrt(M[j, j] - temp)
            else:
                for k in range(j):
                    temp += (L[i, k] * L[j, k])
                if L[j][j] > 0:
                    L[i][j] = (M[i][j] - temp) / L[j][j]

    return L


def solve_Ab(A, b):
    L = cholesky(A)
    N = L.shape[0]
    y = np.zeros((N, 1))
    # Forward pass
    for i in range(N):
        temp = 0
        for j in range(i):
            temp += y[j] * L[i, j]
        if L[i, i] > 0:
            y[i] = (b[i] - temp) / L[i, i]
    # Backward pass
    x = np.zeros((N, 1))
    for i_ in range(N):
        i = N - i_ - 1
        temp = 0
        for j in range(i + 1, N):
            temp += x[j] * L[j, i]
        if L[i, i] > 0:
            x[i] = (y[i] - temp) / L[i, i]
    return x


if __name__ == '__main__':
    sys.exit(main())
