import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为动态欧拉角 (XYZ顺序)
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    if sy > 1e-6:  # 非奇异情况
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:  # 奇异情况
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)
def vector_to_rotation_matrix(a, b):
    """
    从向量a旋转到向量b,返回旋转矩阵
    """
    a = np.array(a)
    b = np.array(b)
    
    # 计算单位向量
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    # 计算旋转轴和角度
    axis = np.cross(a_norm, b_norm)
    axis_norm = np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))

    # 如果旋转轴为零向量，表示不需要旋转
    if axis_norm == 0:
        return np.eye(3)  # 单位矩阵

    # 将旋转轴归一化
    axis = axis / axis_norm

    # 构造旋转矩阵
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    R = np.array([
        [t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - axis[2] * s, t * axis[0] * axis[2] + axis[1] * s],
        [t * axis[1] * axis[0] + axis[2] * s, t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - axis[0] * s],
        [t * axis[2] * axis[0] - axis[1] * s, t * axis[2] * axis[1] + axis[0] * s, t * axis[2] * axis[2] + c]
    ])
    
    return R
def init(a,b):
    

    # 设置坐标轴的标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    # 隐藏网格
    ax.grid(False)

    # 绘制坐标轴
    ax.plot([-5, 5], [0, 0], [0, 0], color='red', linewidth=2)  # X轴
    ax.plot([0, 0], [-5, 5], [0, 0], color='green', linewidth=2)  # Y轴
    ax.plot([0, 0], [0, 0], [-5, 5], color='blue', linewidth=2)  # Z轴

    # 确保坐标轴指向相应的方向
    ax.text(5, 0, 0, 'X', color='red')
    ax.text(0, 5, 0, 'Y', color='green')
    ax.text(0, 0, 5, 'Z', color='blue')

    # 绘制向量
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='black', arrow_length_ratio=0.1)

    # 设置视角
    ax.view_init(elev=20, azim=30)




if __name__ == "__main__":
    # 创建左手坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    vector_a = np.array([0,0,-2])
    vector_b = np.array([1,1,3])
    init(vector_a,vector_b)

    R = vector_to_rotation_matrix(vector_a, vector_b)
    euler_angles = rotation_matrix_to_euler(R)
    print(f"旋转角度 (X, Y, Z): {euler_angles}")
    #x轴旋转视角(z-y)
    #ax.view_init(elev=0, azim=0)
    #y轴旋转视角(z-x)
    ax.view_init(elev=0, azim=90)
    plt.show()