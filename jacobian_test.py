#!/usr/bin/env python3
from pose_graph import J_r_for_matrix, J_for_matrix_numerical
from pose_graph import J_r_for_sync, J_for_sync_numerical

import Lie

import sys
import unittest
import numpy as np
np.set_printoptions(precision=3, linewidth=300,
                    suppress=True, threshold=sys.maxsize)


class TestJacobian(unittest.TestCase):
    def setUp(self):
        self.Z = [Lie.from_rpy_xyz(0., 0., 1., 2, 1, 0),  # T0 + T01 = T1
                  Lie.from_rpy_xyz(0., 1., 0., 2, 1, 0),
                  Lie.from_rpy_xyz(0., 0., 1., 0, 2, 1)
                  ]
        self.noise_Z = [Lie.from_rpy_xyz(0., 0., 1., 2, 1, 0.1),  # T0 + T01 = T1
                        Lie.from_rpy_xyz(0., 1., 0., 2, 1, 0.1),
                        Lie.from_rpy_xyz(0., 0., 1., 0, 2, 1.1)
                        ]
        self.pose_links = [(0, 1),
                           (1, 2),
                           (2, 3)
                           ]
        self.poses_init = [np.eye(4)]
        for z in self.Z:
            self.poses_init.append(self.poses_init[-1].dot(z))
        self.poses_est = [np.eye(4)]
        for z in self.noise_Z:
            self.poses_est.append(self.poses_est[-1].dot(z))

    def test_matrix_jacobian(self):
        J, r = J_r_for_matrix(poses_init=self.poses_init, poses_est=self.poses_est,
                              pose_links=self.pose_links, Z=self.Z, cal_jacobian=True)
        J_n = J_for_matrix_numerical(
            poses_init=self.poses_init, poses_est=self.poses_est, pose_links=self.pose_links, Z=self.Z)
        # row, col = J.shape
        # t_indices = []
        # R_indices = []
        # for idx in range(col // 12):
        #     R_indices.extend(range(idx*12, idx*12+9))
        #     t_indices.extend(range(idx*12+9, (idx+1)*12))
        # self.assertTrue(np.allclose(J[:, t_indices], J_n[:, t_indices]))
        # self.assertTrue(np.allclose(J[:, R_indices], J_n[:, R_indices]))
        self.assertTrue(np.allclose(J, J_n))

    # def test_sync_jacobian(self):
    #     J, r = J_r_for_sync(poses_init=self.poses_init, poses_est=self.poses_est,
    #                           pose_links=self.pose_links, Z=self.Z, cal_jacobian=True)
    #     J_n = J_for_sync_numerical(
    #         poses_init=self.poses_init, poses_est=self.poses_est, pose_links=self.pose_links, Z=self.Z)
    #     self.assertTrue(np.allclose(J, J_n))


if __name__ == '__main__':
    unittest.main()
