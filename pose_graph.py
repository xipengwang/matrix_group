#!/usr/bin/env python3

import Lie
from g2o_reader import read_g2o
import argparse
import sys
import copy
import time

import mpl_toolkits.mplot3d.proj3d as proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import expm, logm
from math import sin, cos, sqrt
import numpy as np
np.set_printoptions(precision=9, linewidth=300,
                    suppress=True, threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=["sync", "matrix"], required=True)
    parser.add_argument(
        '--sigma-scale',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--max-iter',
        default=50,
        type=int
    )
    parser.add_argument(
        '--no-noise',
        help='Odom observation is perfect.',
        action='store_true')
    parser.add_argument(
        '--g2o-file',
    )
    parser.add_argument(
        '--max-id',
        default=20,
        type=int
    )
    args = parser.parse_args()

    np.random.seed(0)
    U = [Lie.from_rpy_xyz(0., 0., 1., 2, 1, 0),  # T0 + T01 = T1
         Lie.from_rpy_xyz(0., 1., 0., 2, 1, 0),
         Lie.from_rpy_xyz(0., 0., 1., 0, 2, 1),
         Lie.from_rpy_xyz(0., 1., 0., -2, -1, 0),
         Lie.from_rpy_xyz(0., 0., 1., -2, -1, 0),
         ]

    poses = [np.eye(4)]
    pose_links = []
    for i, u in enumerate(U):
        poses.append(poses[-1].dot(u))
        pose_links.append((i, i + 1))

    Sigma = np.diag([0.05**2, 0.03**2, 0.03**2, 0.05**2, 0.05**2, 0.05**2])
    L = cholesky(Sigma * args.sigma_scale**2)
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
        (poses_init, Z, pose_links) = read_g2o(args.g2o_file, args.max_id)
        poses = poses_init
    # ax.clear()
    # print(poses_init[0])
    # plot_poses(ax, poses_init, pose_links, 'g')
    # plt.show()
    # return

    # Jacobian
    last_r, cov = init_for_sync(
        odom_Z=odom_Z, Z=Z) if args.mode == "sync" else init_for_matrix(odom_Z=odom_Z, Z=Z)

    iter = 0
    r_eps = 1e-2
    poses_est = copy.deepcopy(poses_init)
    loop_closures = [pose_link for pose_link in pose_links if (pose_link[1] - pose_link[0] != 1)]
    print(f"{len(loop_closures)} loop closures: \n {loop_closures}")

    while iter < args.max_iter:
        print("Iter: ", iter)
        iter += 1
        if args.mode == "sync":
            # J = J_for_sync_numerical(
            #    poses_init=poses_init, poses_est=poses_est, pose_links=pose_links, Z=Z)
            # r = J_r_for_sync(poses_init=poses_init, poses_est=poses_est,
            #                 pose_links=pose_links, Z=Z, cal_jacobian=False)
            J, r = J_r_for_sync(poses_init=poses_init, poses_est=poses_est,
                                pose_links=pose_links, Z=Z, cal_jacobian=True)
            A = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), J])
            b = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), -r])
            dx = solve_Ab(A, b)
            # dx = np.linalg.inv(A).dot(b)
            for i in range(len(poses_est)):
                poses_est[i] = poses_est[i].dot(
                    expm(Lie.hat(dx[i * 6:(i + 1) * 6])))
            print("residual: ", np.linalg.norm(r))
        else:
            # J = J_for_matrix_numerical(
            #    poses_init=poses_init, poses_est=poses_est, pose_links=pose_links, Z=Z)
            # r = J_r_for_matrix(poses_init=poses_init, poses_est=poses_est,
            #                   pose_links=pose_links, Z=Z, cal_jacobian=False)
            # start = time.time()
            J, r = J_r_for_matrix(poses_init=poses_init, poses_est=poses_est,
                                  pose_links=pose_links, Z=Z, cal_jacobian=True)
            # print(time.time() - start)
            A = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), J])
            b = np.linalg.multi_dot([J.transpose(), np.linalg.inv(cov), -r])
            # print(time.time() - start)
            dx = solve_Ab(A, b)
            # print(time.time() - start)
            for i in range(len(poses_est)):
                poses_est[i][0:3, 0:3] = poses_est[i][0:3, 0:3] + \
                    dx[12 * i:12 * i + 9].reshape(3, 3)
                # Polarization
                U, _, V = np.linalg.svd(
                    poses_est[i][0:3, 0:3], full_matrices=True)
                S = np.eye(3)
                S[2, 2] = 1 if np.linalg.det(U) * np.linalg.det(V) > 0 else -1
                poses_est[i][0:3, 0:3] = U.dot(S).dot(V)
                poses_est[i][0:3, [3]] = poses_est[i][0:3, [3]] + \
                    dx[12 * i + 9:12 * i + 12]
            print("residual: ", np.linalg.norm(r))

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
    plt.show()


def init_for_sync(*, odom_Z, Z):
    last_r = np.zeros((6 + 6 * len(Z), 1))
    cov = np.eye(6 + 6 * len(Z))
    # Set supper small variance for first measurement
    cov[0:6, 0:6] *= 0.000001
    # Trust more of loop closures
    cov[6 * len(odom_Z):6 + 6 * len(Z), 6 * len(odom_Z):6 + 6 * len(Z)] *= 0.01
    return (last_r, cov)


def J_for_sync_numerical(*, poses_init, poses_est, pose_links, Z):
    J = np.zeros((6 + 6 * len(Z), 6 * len(poses_init)))
    kDelta = 0.00001
    for col in range(J.shape[1]):
        pose_idx = col // 6
        param_idx = col % 6
        new_poses_est = copy.deepcopy(poses_est)
        eta = np.zeros((6, 1))
        eta_plus = copy.deepcopy(eta)
        eta_plus[param_idx] += kDelta
        eta_minus = copy.deepcopy(eta)
        eta_minus[param_idx] -= kDelta
        new_poses_est[pose_idx] = poses_est[pose_idx].dot(
            expm(Lie.hat(eta_plus)))
        r_plus = J_r_for_sync(poses_init=poses_init, poses_est=new_poses_est,
                              pose_links=pose_links, Z=Z, cal_jacobian=False)
        new_poses_est[pose_idx] = poses_est[pose_idx].dot(
            expm(Lie.hat(eta_minus)))
        r_minus = J_r_for_sync(poses_init=poses_init, poses_est=new_poses_est,
                               pose_links=pose_links, Z=Z, cal_jacobian=False)
        J[:, [col]] = (r_plus - r_minus) / (2 * kDelta)
    return J


def J_r_for_sync(*, poses_init, poses_est, pose_links, Z, cal_jacobian):
    J = np.zeros((6 + 6 * len(Z), 6 * len(poses_init)))
    r = np.zeros((6 + 6 * len(Z), 1))
    r[0:6] = Lie.vee(logm(np.linalg.inv(poses_init[0]).dot(poses_est[0])))
    J[0:6, 0:6] = Lie.RightJacobianInverse_SE3(r[0:6])
    for i, (a, b) in enumerate(pose_links):
        # Measurement is T_ab = T_a.inv() * T_b
        # ||J*dx + h(x) - z||_2
        # r = h(x) - z = vee(log(T_a.inv()*T_b)) - vee(log(T_ab))
        # r = vee(log(T_a.inv()*T_b)) - vee(log(T_ab))
        # r = vee(log(T_ba * (T_a.inv()*T_b)))
        res_idx = 6 * (i + 1)

        R_ab_z = Z[i][:3, :3]
        t_ab_z = Z[i][:3, [3]]
        R_ba_z = R_ab_z.transpose()
        t_ba_z = -R_ab_z.transpose().dot(t_ab_z)
        T_a = poses_est[a]
        T_b = poses_est[b]
        R_a = T_a[:3, :3]
        R_b = T_b[:3, :3]
        t_a = T_a[0:3, [3]]
        t_b = T_b[0:3, [3]]
        R_ab = R_a.transpose().dot(R_b)
        tb_minus_ta = t_b - t_a
        t_ab = R_a.transpose().dot(tb_minus_ta)

        # res = Lie.vee(np.linalg.inv(Z[i]).dot(
        #     np.linalg.inv(poses_est[a]).dot(poses_est[b])))
        r_T = np.zeros((4, 4))
        r_T[3, 3] = 1
        r_T[:3, :3] = R_ba_z.dot(R_a.transpose()).dot(R_b)
        r_T[:3, [3]] = R_ba_z.dot(R_a.transpose()).dot(tb_minus_ta) + t_ba_z
        res = Lie.vee(r_T)

        r[res_idx:res_idx + 6] = res
        if not cal_jacobian:
            continue
        # Measurement is T_ab = T_a.inv() * T_b
        T_ba = np.zeros((4, 4))
        T_ba[3, 3] = 1
        T_ba[:3, :3] = R_b.transpose().dot(R_a)
        T_ba[:3, [3]] = R_b.transpose().dot(t_a - t_b)
        J[res_idx:res_idx + 6, a * 6:(a + 1) * 6] = - \
            Lie.RightJacobianInverse_SE3(res).dot(Lie.Adjoint_SE3(T_ba))
        J[res_idx:res_idx + 6, b *
          6:(b + 1) * 6] = Lie.RightJacobianInverse_SE3(res)
        # solve normal equations
    if not cal_jacobian:
        return r
    else:
        return (J, r)


def init_for_matrix(*, odom_Z, Z):
    last_r = np.zeros((12 + 12 * len(Z), 1))
    cov = np.eye(12 + 12 * len(Z))
    # Set supper small variance for first measurement
    cov[0:12, 0:12] *= 0.000001
    # Trust more of loop closures
    cov[12 * len(odom_Z):12 + 12 * len(Z), 12 *
        len(odom_Z):12 + 12 * len(Z)] *= 0.01
    return (last_r, cov)


def J_for_matrix_numerical(*, poses_init, poses_est, pose_links, Z):
    J = np.zeros(((len(Z) + 1) * 12, len(poses_init) * 12))
    kDelta = 0.00001
    for col in range(J.shape[1]):
        pose_idx = col // 12
        pose_col = col % 12
        if pose_col < 9:
            i = pose_col // 3
            j = pose_col % 3
        else:
            i = pose_col % 3
            j = 3
        new_poses_est_plus = copy.deepcopy(poses_est)
        new_poses_est_plus[pose_idx][i, j] = poses_est[pose_idx][i, j] + kDelta
        r_plus = J_r_for_matrix(poses_init=poses_init, poses_est=new_poses_est_plus,
                                pose_links=pose_links, Z=Z, cal_jacobian=False)
        new_poses_est_minus = copy.deepcopy(poses_est)
        new_poses_est_minus[pose_idx][i, j] = poses_est[pose_idx][i, j] - kDelta
        r_minus = J_r_for_matrix(poses_init=poses_init, poses_est=new_poses_est_minus,
                                 pose_links=pose_links, Z=Z, cal_jacobian=False)
        J[:, [col]] = (r_plus - r_minus) / (2 * kDelta)
    return J


def J_r_for_matrix(*, poses_init, poses_est, pose_links, Z, cal_jacobian):
    J = np.zeros(((len(Z) + 1) * 12, len(poses_init) * 12))
    r = np.zeros(((len(Z) + 1) * 12, 1))
    r[0:9] = (poses_est[0][0:3, 0:3] - np.eye(3)).reshape(9, 1)
    r[9:12] = poses_est[0][0:3, [3]] - poses_init[0][0:3, [3]]
    J[:12, :12] = np.eye(12)
    for i, (a, b) in enumerate(pose_links):
        res_idx = 12 * (i + 1)
        R_ab_z = Z[i][:3, :3]
        t_ab_z = Z[i][:3, [3]]
        T_a = poses_est[a]
        T_b = poses_est[b]
        R_a = T_a[:3, :3]
        R_b = T_b[:3, :3]
        t_a = T_a[0:3, [3]]
        t_b = T_b[0:3, [3]]
        R_ab = R_a.transpose().dot(R_b)
        tb_minus_ta = t_b - t_a
        t_ab = R_a.transpose().dot(tb_minus_ta)
        # T_ab = np.linalg.inv(T_a).dot(T_b)
        # assert(np.allclose(t_ab, T_ab[:3, [3]]))
        # assert(np.allclose(R_ab, T_ab[:3, :3]))
        r[res_idx:res_idx +
          9] = (R_ab - R_ab_z).reshape(9, 1)
        r[res_idx + 9:res_idx + 12] = t_ab - t_ab_z
        if not cal_jacobian:
            continue

        # J for pose_a.
        base_a = a * 12
        t_idx = range(base_a + 9, base_a + 12)
        J[res_idx: res_idx + 3, [base_a + 0]] = R_b[[0], :].transpose()
        J[res_idx: res_idx + 3, [base_a + 3]] = R_b[[1], :].transpose()
        J[res_idx: res_idx + 3, [base_a + 6]] = R_b[[2], :].transpose()

        J[res_idx + 3: res_idx + 6, [base_a + 1]] = R_b[[0], :].transpose()
        J[res_idx + 3: res_idx + 6, [base_a + 4]] = R_b[[1], :].transpose()
        J[res_idx + 3: res_idx + 6, [base_a + 7]] = R_b[[2], :].transpose()

        J[res_idx + 6: res_idx + 9, [base_a + 2]] = R_b[[0], :].transpose()
        J[res_idx + 6: res_idx + 9, [base_a + 5]] = R_b[[1], :].transpose()
        J[res_idx + 6: res_idx + 9, [base_a + 8]] = R_b[[2], :].transpose()

        J[res_idx + 9 : res_idx + 12, base_a: base_a + 3] = tb_minus_ta[0, 0] * np.eye(3)
        J[res_idx + 9 : res_idx + 12, base_a + 3: base_a + 6] = tb_minus_ta[1, 0] * np.eye(3)
        J[res_idx + 9 : res_idx + 12, base_a + 6: base_a + 9] = tb_minus_ta[2, 0] * np.eye(3)

        # t_a
        J[res_idx + 9:res_idx + 12, t_idx] = -R_a.transpose()

        # J for pose_b.
        base_b = b * 12
        t_idx = range(base_b + 9, base_b + 12)
        R_bg_z = R_a.transpose()

        J[res_idx: res_idx + 3, base_b: base_b + 3] = R_bg_z[0, 0] * np.eye(3)
        J[res_idx: res_idx + 3, base_b + 3: base_b + 6] = R_bg_z[0, 1] * np.eye(3)
        J[res_idx: res_idx + 3, base_b + 6: base_b + 9] = R_bg_z[0, 2] * np.eye(3)

        J[res_idx + 3: res_idx + 6, base_b: base_b + 3] = R_bg_z[1, 0] * np.eye(3)
        J[res_idx + 3: res_idx + 6, base_b + 3: base_b + 6] = R_bg_z[1, 1] * np.eye(3)
        J[res_idx + 3: res_idx + 6, base_b + 6: base_b + 9] = R_bg_z[1, 2] * np.eye(3)

        J[res_idx + 6: res_idx + 9, base_b: base_b + 3] = R_bg_z[2, 0] * np.eye(3)
        J[res_idx + 6: res_idx + 9, base_b + 3: base_b + 6] = R_bg_z[2, 1] * np.eye(3)
        J[res_idx + 6: res_idx + 9, base_b + 6: base_b + 9] = R_bg_z[2, 2] * np.eye(3)

        J[res_idx + 9:res_idx + 12, t_idx] = R_a.transpose()
    if cal_jacobian:
        return (J, r)
    else:
        return r


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
