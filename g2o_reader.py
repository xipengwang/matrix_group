#!/usr/bin/env python3
import os
import argparse
import numpy as np

def read_g2o(file_path):
    print("reading " + file_path)
    vertex = 0
    edge = 0
    with open(file_path, "r") as f:
        poses_init = []
        Z = []
        pose_links = []
        MAX_ID = 10
        for line in f:
            entries = line.split(" ")
            if vertex == 0 and entries[0] == 'VERTEX_SE3:QUAT':
                ID = int(entries[1])
                x = float(entries[2])
                y = float(entries[3])
                z = float(entries[4])
                qx = float(entries[5])
                qy = float(entries[6])
                qz = float(entries[7])
                qw = float(entries[8])
                pose = np.array([[1. - 2.*qy*qy - 2.*qz*qz, 2.*qx*qy - 2.*qz*qw, 2.*qx*qz + 2.*qy*qw, x],
                                  [2.*qx*qy + 2.*qz*qw, 1. - 2.*qx*qx - 2.*qz*qz, 2.*qy*qz - 2.*qx*qw, y],
                                  [2.*qx*qz - 2.*qy*qw, 2.*qy*qz + 2.*qx*qw, 1. - 2.*qx*qx - 2.*qy*qy, z],
                                  [0., 0., 0., 1.]])
                if ID <= MAX_ID:
                    poses_init.append(pose)
            if edge == 0 and entries[0] == 'EDGE_SE3:QUAT':
                a = int(entries[1])
                b = int(entries[2])
                x = float(entries[3])
                y = float(entries[4])
                z = float(entries[5])
                qx = float(entries[6])
                qy = float(entries[7])
                qz = float(entries[8])
                qw = float(entries[9])
                T_ab = np.array([[1. - 2.*qy*qy - 2.*qz*qz, 2.*qx*qy - 2.*qz*qw, 2.*qx*qz + 2.*qy*qw, x],
                                 [2.*qx*qy + 2.*qz*qw, 1. - 2.*qx*qx - 2.*qz*qz, 2.*qy*qz - 2.*qx*qw, y],
                                 [2.*qx*qz - 2.*qy*qw, 2.*qy*qz + 2.*qx*qw, 1. - 2.*qx*qx - 2.*qy*qy, z],
                                 [0., 0., 0., 1.]])
                if a<=MAX_ID and b<=MAX_ID:
                    pose_links.append((a,b))
                    Z.append(T_ab)
        return (poses_init, Z, pose_links)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--g2o-file',
        required=True
    )
    args = parser.parse_args()
    read_g2o(args.g2o_file)


if __name__ == '__main__':
    main()
