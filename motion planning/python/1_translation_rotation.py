from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np


# 四元数(q1,q2,q3,w)->欧拉角(x,y,z)
def quaterntion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler


# 四元数(q1,q2,q3,w)->旋转矩阵
def quaterntion2rotation(quaternion):
    r = R.from_quat(quaternion)
    rotation = r.as_matrix()
    return rotation


# 欧拉角(x,y,z)->四元数(q1,q2,q3,w)
def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion


# 欧拉角(x,y,z)->旋转矩阵
def euler2rotation(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation = r.as_matrix()
    return rotation


# 旋转矩阵->四元数(q1,q2,q3,w)
def rotation2quaternion(rotation):
    r = R.from_matrix(rotation)
    quaternion = r.as_quat()
    return quaternion


# 旋转矩阵->欧拉角(x,y,z)
def rotation2euler(rotation):
    r = R.from_matrix(rotation)
    euler = r.as_euler('xyz', degrees=True)
    return euler


# 旋转矩阵->旋转向量
def rotation2vector(rotation):
    vector, j = cv2.Rodrigues(rotation)  # 当输入量为旋转矩阵时，自动返回旋转向量和雅可比矩阵
    return vector


# 旋转向量->旋转矩阵
def vector2rotation(vector):
    rotation, j = cv2.Rodrigues(vector)  # 当输入量为旋转向量时，自动返回旋转矩阵和雅可比矩阵
    return rotation


if __name__ == "__main__":
    # 测试代码
    rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    euler = [0, 0, 0]
    vector = np.array([0.2, 0.4, 0.6])
    print(type(vector2rotation(vector)))
    # print(rotation2quaternion(rotation))
    # print(euler2rotation(euler))
    print(vector2rotation(vector))
    print(rotation2vector(vector2rotation(vector)))
