import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_rotation_matrix(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    rotation_axis = np.cross(a_norm, b_norm)
    rotation_angle = np.arccos(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))
    
    if np.linalg.norm(rotation_axis) < 1e-6:
        return np.eye(3)

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    R_mat = R.from_rotvec(rotation_axis * rotation_angle).as_matrix()
    return R_mat

def rotation_matrix_to_euler(R_mat):
    euler_angles = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
    return euler_angles[:2]  # 只返回 X 和 Y

# 示例向量
a = [0, 0, -2]
b = [1, 1, 1]

R_mat = calculate_rotation_matrix(np.array(a), np.array(b))
euler_angles = rotation_matrix_to_euler(R_mat)
print(f"旋转角度 (X, Y): {euler_angles}")
