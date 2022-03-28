# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.spatial.transform import Rotation as R


# 旋转矩阵->旋转矢量
def rm2rv(R):
    theta = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    K = (1 / (2 * np.sin(theta))) * np.asarray([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
    r = theta * K
    r1 = [R[0][3], R[1][3], R[2][3], r[0], r[1], r[2]]
    return r1


# 旋转矢量->旋转矩阵
def rv2rm(rv):
    rx = rv[3]
    ry = rv[4]
    rz = rv[5]
    theta = np.linalg.norm([rx, ry, rz])
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta

    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c

    R = np.zeros((4, 4))
    R[0][0] = kx * kx * v + c
    R[0][1] = kx * ky * v - kz * s
    R[0][2] = kx * kz * v + ky * s
    R[0][3] = rv[0]

    R[1][0] = ky * kx * v + kz * s
    R[1][1] = ky * ky * v + c
    R[1][2] = ky * kz * v - kx * s
    R[1][3] = rv[1]

    R[2][0] = kz * kx * v - ky * s
    R[2][1] = kz * ky * v + kx * s
    R[2][2] = kz * kz * v + c
    R[2][3] = rv[2]
    R[3][3] = 1

    return R


# 旋转矩阵->rpy欧拉角
def rm2rpy(R):
    # sy = np.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    sy = np.sqrt(R[2][1] * R[2][1] + R[2][2] * R[2][2])
    singular = sy < 1e-6

    if not singular:
        rotatex = np.arctan2(R[2][1], R[2][2])
        rotatey = np.arctan2(-R[2][0], sy)
        rotatez = np.arctan2(R[1][0], R[0][0])
    else:
        rotatex = np.arctan2(-R[1][2], R[1][1])
        rotatey = np.arctan2(-R[2][0], sy)
        rotatez = 0

    return [R[0][3], R[1][3], R[2][3], rotatex, rotatey, rotatez]


# rpy->旋转矩阵
def rpy2rm(rpy):
    # # Rx = np.zeros((3, 3), dtype=rpy.dtype)
    # # Ry = np.zeros((3, 3), dtype=rpy.dtype)
    # # Rz = np.zeros((3, 3), dtype=rpy.dtype)
    #
    # R0 = np.zeros((4, 4))
    #
    # x = rpy[0]
    # y = rpy[1]
    # z = rpy[2]
    # thetaX = rpy[3]
    # thetaY = rpy[4]
    # thetaZ = rpy[5]
    #
    # cx = np.cos(thetaX)
    # sx = np.sin(thetaX)
    #
    # cy = np.cos(thetaY)
    # sy = np.sin(thetaY)
    #
    # cz = np.cos(thetaZ)
    # sz = np.sin(thetaZ)
    #
    # R0[0][0] = cz * cy
    # R0[0][1] = cz * sy * sx - sz * cx
    # R0[0][2] = cz * sy * cx + sz * sx
    # R0[0][3] = x
    # R0[1][0] = sz * cy
    # R0[1][1] = sz * sy * sx + cz * cx
    # R0[1][2] = sz * sy * cx - cz * sx
    # R0[1][3] = y
    # R0[2][0] = -sy
    # R0[2][1] = cy * sx
    # R0[2][2] = cy * cx
    # R0[2][3] = z
    # R0[3][3] = 1
    # print(R0)
    # return R0

    rm1 = R.from_euler('xyz', rpy[3:], degrees=True)
    rm2 = rm1.as_matrix()
    rm3 = np.insert(rm2, 3, values=rpy[:3], axis=1)
    rm = np.insert(rm3, 3, values=[0, 0, 0, 1], axis=0)
    return rm


def rv2rpy(rv):
    R = rv2rm(rv)
    rpy = rm2rpy(R)
    return rpy


def rpy2rv(rpy):
    R = rpy2rm(rpy)
    rv = rm2rv(R)
    return rv


def rpy2qt(rpy):
    rx = rpy[3]
    ry = rpy[4]
    rz = rpy[5]
    x = np.cos(ry / 2) * np.cos(rz / 2) * np.sin(rx / 2) - np.sin(ry / 2) * np.sin(rz / 2) * np.cos(rx / 2)
    y = np.sin(ry / 2) * np.cos(rz / 2) * np.cos(rx / 2) + np.cos(ry / 2) * np.sin(rz / 2) * np.sin(rx / 2)
    z = np.cos(ry / 2) * np.sin(rz / 2) * np.cos(rx / 2) - np.sin(ry / 2) * np.cos(rz / 2) * np.sin(rx / 2)
    w = np.cos(ry / 2) * np.cos(rz / 2) * np.cos(rx / 2) + np.sin(ry / 2) * np.sin(rz / 2) * np.sin(rx / 2)
    return [rpy[0], rpy[1], rpy[2], w, x, y, z]


def qt2rpy(qt):
    Rx = qt[0]
    Ry = qt[1]
    Rz = qt[2]
    w = qt[3]
    x = qt[4]
    y = qt[5]
    z = qt[6]

    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(2 * (w * y - z * x))
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    r = math.degrees(r)
    p = math.degrees(p)
    y = math.degrees(y)
    return [Rx, Ry, Rz, r, p, y]


def rm2qt(rm):
    rm_angle = np.array(rm)[0:3, 0:3]
    r1 = R.from_matrix(rm_angle)
    r = r1.as_quat()
    return [rm[0, 3], rm[1, 3], rm[2, 3], r[3], r[0], r[1], r[2]]


def qt2rm(qt):
    qt_angle = np.insert(qt[4:], 3, qt[3])
    rm1 = R.from_quat(qt_angle)
    rm2 = rm1.as_matrix()
    rm3 = np.insert(rm2, 3, values=qt[:3], axis=1)
    rm = np.insert(rm3, 3, values=[0, 0, 0, 1], axis=0)
    return rm
