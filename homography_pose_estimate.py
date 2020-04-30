#!/usr/bin/env python3

from scipy.linalg import expm, logm
import math
import Lie
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.proj3d as proj3d
import cv2 as cv
import argparse
import sys
import os

from objloader_simple import *
import numpy as np
np.set_printoptions(precision=3, linewidth=300,
                    suppress=True, threshold=sys.maxsize)


dir_name = os.getcwd()
obj = OBJ(os.path.join(dir_name, './models/wolf.obj'), swapyz=True)
use_perfect_detector = True
kX = 3.0
kY = -6.0
kZ = 6.0
kTileWidth = 0.5
kCheckerBoardSize = 0.5
kImageWidth = 64 * 10
kImageHeight = 32 * 10
fx = 600
kIntrinsic = np.array(
    [[fx, 0, kImageWidth / 2], [0, fx, kImageHeight / 2], [0, 0, 1]])
kIntrinsicInv = np.linalg.inv(kIntrinsic)


def main():
    _tests()
    # camera_pose_ideal = Lie.from_rpy_xyz(math.pi, 0, 0., kX, kY, kZ)
    camera_pose_ideal = Lie.from_rpy_xyz(0, 0, 0., kX, kY, kZ)
    angle = math.acos(-kZ / math.sqrt(kX*kX + kY*kY + kZ*kZ))
    rotation_vector = np.array([[kY], [-kX], [0]])
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    if rotation_vector_norm < 1e-6:
        rotation_vector[0] = angle
        rotation_vector[1] = 0
        rotation_vector[2] = 0
    else:
        rotation_vector = rotation_vector / rotation_vector_norm * angle
    rotation_matrix = expm(Lie.skew(rotation_vector))
    Sigma = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1**2, 0.1**2, 0.1**2])
    L = cholesky(Sigma)
    noise = L.dot(np.random.normal(0, 1, (6, 1)))
    camera_pose = camera_pose_ideal
    camera_pose[0:3, 0:3] = rotation_matrix
    # camera_pose = camera_pose.dot(expm(Lie.hat(noise)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    rect0 = FancyBboxPatch((0, 0), kTileWidth, kTileWidth,
                           'square', fill=True, color='k')
    ax.add_patch(rect0)
    pathpatch_2d_to_3d(rect0, z=0, normal='z')
    pathpatch_translate(rect0, (kTileWidth, kTileWidth, 0))
    rect1 = FancyBboxPatch((0, 0), kTileWidth, kTileWidth,
                           'square', fill=True, color='k')
    ax.add_patch(rect1)
    pathpatch_2d_to_3d(rect1, z=0, normal='z')
    pathpatch_translate(rect1, (-kTileWidth, -kTileWidth, 0))

    camera_pose_t = camera_pose[0:3, [3]]
    camera_pose_R = camera_pose[0:3, 0:3]
    image = np.zeros((kImageHeight, kImageWidth, 1), np.uint8)
    for x in range(kImageWidth):
        for y in range(kImageHeight):
            # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
            # normal n = (0, 0, 1)
            ray_c = kIntrinsicInv.dot(
                np.array([[x], [y], [1]]))
            ray_w = camera_pose_R.dot(ray_c)
            alpha = -camera_pose_t[2] / ray_w[2]
            assert alpha > 0, "It is behind camera!"
            p = camera_pose_t + alpha * ray_w
            col = math.floor(p[0] / kTileWidth)
            row = math.floor(p[1] / kTileWidth)
            if not _check_black_or_with(row, col) or abs(p[0]) > kCheckerBoardSize or abs(p[1]) > kCheckerBoardSize:
                # white
                image[y][x] = 255
    blur = cv.blur(image, (3, 3))
    corners = cv.goodFeaturesToTrack(blur, 25, 0.01, 10)
    corners = np.int0(corners)
    pts_src, pts_dst, image_corners = _hack_way_finding_matches(corners)

    # Perfect detector
    if use_perfect_detector:
        pts_dst = []
        for i, p in enumerate(pts_src):
            p_c = np.linalg.inv(camera_pose).dot(
                np.array([[pts_src[i][0]], [pts_src[i][1]], [0], [1]], dtype=float))
            pixel_homo = kIntrinsic.dot(p_c[0:3, :])
            pixel = pixel_homo[0:2, :] / pixel_homo[2]
            pts_dst.append([pixel[0], pixel[1]])
        pts_dst = np.array(pts_dst)
    H, status = cv.findHomography(pts_src, pts_dst)

    Rt = kIntrinsicInv.dot(H)
    len1 = np.linalg.norm(Rt[:, [0]])
    len2 = np.linalg.norm(Rt[:, [1]])
    if Rt[2][2] < 0:
        Rt *= -1
    Rt /= 0.5*(len1 + len2)
    pose_c_tag = np.zeros((4, 4))
    pose_c_tag[0:3, [3]] = Rt[:, [2]]
    pose_c_tag[3][3] = 1
    pose_c_tag[0:3, 0:2] = Rt[:, 0:2]
    pose_c_tag[0:3, [2]] = Lie.skew(Rt[:, [0]]).dot(Rt[:, [1]])
    R = pose_c_tag[0:3, 0:3]
    u, _, vh = np.linalg.svd(R)
    pose_c_tag[0:3, 0:3] = u.dot(vh)
    recovered_camera_pose = np.linalg.inv(pose_c_tag)

    draw_camera_pose(ax, camera_pose, 'r')
    print('error so3: \n', Lie.vee(logm(pose_c_tag.dot(camera_pose))))
    # print(camera_pose)
    # print(recovered_camera_pose)
    print("Re-projection error")
    err = 0
    for i, p in enumerate(pts_src):
        p_c = np.linalg.inv(recovered_camera_pose).dot(
            np.array([[pts_src[i][0]], [pts_src[i][1]], [0], [1]], dtype=float))
        pixel_homo = kIntrinsic.dot(p_c[0:3, :])
        pixel = pixel_homo[0:2, :] / pixel_homo[2]
        e = (pixel[0] - pts_dst[i][0]) ** 2 + (pixel[1] - pts_dst[i][1]) ** 2
        err += e
    mean_err = err / i
    print('mean re-projection error', mean_err)

    draw_camera_pose(ax, recovered_camera_pose, 'k')
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
    plt.show()

    cv.imshow('CheckerBoardImage', image)
    cv.waitKey(0)
    render(image_corners, obj, kIntrinsic.dot(pose_c_tag[0:3, :]), color=False)
    cv.imshow('CheckerBoardCorners', image_corners)
    cv.waitKey(0)
    cv.destroyAllWindows()


def _hack_way_finding_matches(corners):
    image_corners = np.zeros((kImageHeight, kImageWidth, 3), np.uint8)
    clusters = []
    for i in corners:
        x, y = i.ravel()
        if not clusters:
            clusters.append([(x, y)])
        else:
            new_cluster = True
            for cluster in clusters:
                if abs(cluster[0][1] - y) < 5:
                    cluster.append((x, y))
                    new_cluster = False
                    break
            if new_cluster:
                clusters.append([(x, y)])
        image_corners[y][x] = 255
    clusters.sort(key=lambda cluster: cluster[0][1])
    middle_point_idx = -1
    for i, cluster in enumerate(clusters):
        cluster.sort(key=lambda point: point[0])
        if len(cluster) == 3:
            middle_point_idx = i
    # Tag as source.
    pts_src = np.array([[0, kTileWidth], [kTileWidth, kTileWidth],
                        [-kTileWidth, 0], [0, 0], [kTileWidth, 0],
                        [-kTileWidth, -kTileWidth], [0, -kTileWidth]])
    pts_dst = []
    for i in [0, middle_point_idx, -1]:
        for (x, y) in clusters[i]:
            pts_dst.append([x, y])
    pts_dst = np.array(pts_dst)
    return pts_src, pts_dst, image_corners


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def _plot_line(ax, T_w_c, x, y, z, c):
    a = np.matmul(T_w_c, np.array([x[0], y[0], z[0], 1]))
    b = np.matmul(T_w_c, np.array([x[1], y[1], z[1], 1]))
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], c)


def draw_camera_pose(ax, T_w_c, c):
    # Camera is looking at positive z
    kAxisLen = 1
    kHalfAxisLen = kAxisLen / 2.0
    x = np.matmul(T_w_c, np.array([kAxisLen, 0, 0, 1]))
    y = np.matmul(T_w_c, np.array([0, kAxisLen, 0, 1]))
    z = np.matmul(T_w_c, np.array([0, 0, kAxisLen, 1]))
    x_axis = Arrow3D([T_w_c[0, 3], x[0]], [T_w_c[1, 3], x[1]], [T_w_c[2, 3], x[2]], mutation_scale=20,
                     lw=2, arrowstyle="-|>", color='r')
    y_axis = Arrow3D([T_w_c[0, 3], y[0]], [T_w_c[1, 3], y[1]], [T_w_c[2, 3], y[2]], mutation_scale=20,
                     lw=2, arrowstyle="-|>", color='g')
    z_axis = Arrow3D([T_w_c[0, 3], z[0]], [T_w_c[1, 3], z[1]], [T_w_c[2, 3], z[2]], mutation_scale=20,
                     lw=2, arrowstyle="-|>", color='b')
    ax.add_artist(x_axis)
    ax.add_artist(y_axis)
    ax.add_artist(z_axis)
    _plot_line(ax, T_w_c,
               [-kHalfAxisLen, -kHalfAxisLen], [-kHalfAxisLen, kHalfAxisLen], [kAxisLen, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [-kHalfAxisLen, kHalfAxisLen], [kHalfAxisLen, kHalfAxisLen], [kAxisLen, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [kHalfAxisLen, kHalfAxisLen], [kHalfAxisLen, -kHalfAxisLen], [kAxisLen, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [kHalfAxisLen, -kHalfAxisLen], [-kHalfAxisLen, -kHalfAxisLen], [kAxisLen, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [0, -kHalfAxisLen], [0, -kHalfAxisLen], [0, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [0, -kHalfAxisLen], [0, kHalfAxisLen], [0, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [0, kHalfAxisLen], [0, kHalfAxisLen], [0, kAxisLen], c)
    _plot_line(ax, T_w_c,
               [0, kHalfAxisLen], [0, -kHalfAxisLen], [0, kAxisLen], c)


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
                L[j, j] = math.sqrt(M[j, j] - temp)
            else:
                for k in range(j):
                    temp += (L[i, k] * L[j, k])
                if L[j][j] > 0:
                    L[i][j] = (M[i][j] - temp) / L[j][j]

    return L


def _check_black_or_with(row, col):
    # (0, 0) is a black square
    if row % 2 == 0:
        state = True
    else:
        state = False
    if col % 2 != 0:
        state = not state
    return state


def _tests():
    assert _check_black_or_with(0, 0) == True
    assert _check_black_or_with(0, 1) == False
    assert _check_black_or_with(1, 0) == False
    assert _check_black_or_with(1, 1) == True
    assert _check_black_or_with(-1, -1) == True
    assert _check_black_or_with(0, -1) == False
    assert _check_black_or_with(-1, 0) == False


def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[0,  d[2],  -d[1]],
                     [-d[2],     0,  d[0]],
                     [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array(
        [np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

def render(img, obj, projection, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.001

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0], p[1], p[2]] for p in points])
        dst = cv.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv.fillConvexPoly(img, imgpts, color)

    return img

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

if __name__ == '__main__':
    sys.exit(main())
