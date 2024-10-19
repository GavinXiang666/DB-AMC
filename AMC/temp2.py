
import math
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json,csv
from datetime import datetime 
'''
创建程序日志:
    vector_rotate.log--->向量旋转测试log
'''
logger.add("AMC/log/vector_rotate.log",rotation="10 MB")
logger.info(f"Start Time: {datetime.now()}") 

#创建骨骼旋转文件
with open('AMC/output/bone_rotate_angle.csv', 'w', newline='') as angle_file:
    csv_writer = csv.writer(angle_file)
    #angle_file.close()

def write_rotate_angle(bone_id,rotate_angle):
    #追加模式(a),防止覆盖
    with open('AMC/output/bone_rotate_angle.csv', 'a', newline='') as angle_file:
        csv_writer = csv.writer(angle_file)
        csv_writer.writerow([bone_id,rotate_angle])

def read_pose_data(file_path):
    with open(file_path, 'r') as pose_data_json:
        pose_data = json.load(pose_data_json)
        pose_data_json.close()
    return pose_data

#坐标轴AX类型
class Ax:
    def __init__(self,layout,fig,projection=None):
        '''
        layout--->布局,figure的排列顺序
        fig--->figure
        projection--->'2d'和'3d'两种类型(默认None为2d模式)
        '''
        self.projection = projection
        self.layout = layout
        self.fig = fig
        self.ax = self.fig.add_subplot(self.layout, projection=self.projection)
        self.ax.set_xlim(-400, 400)  
        self.ax.set_ylim(-400, 400)  
        #设置坐标系与OpenCV一致
        if self.projection is None:
            #self.ax.invert_yaxis()
            self.ax.plot([-400, 400], [0, 0], [0, 0], color='red', linewidth=2)#X轴
            self.ax.plot([0, 0], [-400, 400], [0, 0], color='green', linewidth=2)#Y轴
        else:
            self.ax.set_zlim(-400, 400)
            
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            # 隐藏网格
            self.ax.grid(False)

            # 绘制坐标轴
            self.ax.plot([-400, 400], [0, 0], [0, 0], color='red', linewidth=2)#X轴
            self.ax.plot([0, 0], [-400, 400], [0, 0], color='green', linewidth=2)#Y轴
            self.ax.plot([0, 0], [0, 0], [-400, 400], color='blue', linewidth=2)#Z轴

            self.ax.text(100, 0, 0, 'X', color='red')
            self.ax.text(0, 100, 0, 'Y', color='green')
            self.ax.text(0, 0, 100, 'Z', color='blue')
            #调整视角
            self.ax.view_init(elev=0, azim=0,roll=0)
    #设置窗口标题
    def set_title(self,title_name):
        self.ax.set_title(title_name)
    #在图像上绘制点
    def draw_scatter(self,scatter_list,color):
        for scatter in scatter_list:
            try:
                x, y, z = scatter
                self.ax.scatter(x,y,z,s=2,color=color)
            except:
                x, y = scatter
                self.ax.scatter(x,y,s=2,color=color)

    #在图像上绘制二维向量
    def draw_2d_vector(self,vector,color):
        self.ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1,color=color)
        
    #在图像上绘制三维向量
    def draw_3d_vector(self,vector,color):
        self.ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, arrow_length_ratio=0.1)

    

#初始化画布figure
def init_figure():
    #3D画布
    global initialize_ax_3d
    initialize_fig = plt.figure(num="initialize pose",figsize=(8, 8))
    initialize_ax_3d = Ax(111,initialize_fig,'3d')
    initialize_ax_3d.set_title('Initialize 3D')
    #X-Z画布
    global initialize_xz
    xz_figure = plt.figure(num="initialize x-z",figsize=(4, 4))
    initialize_xz = Ax(111,xz_figure)
    #X-Y画布
    global initialize_xy
    xy_figure = plt.figure(num="initialize x-y",figsize=(4, 4))
    initialize_xy = Ax(111,xy_figure)

#夹角公式,给定两条直线的斜率k后用反正切函数计算角度
def calculate_angle(k1, k2):
    #判断除数是否为0
    if (1 + k1 * k2) != 0:
        tan_theta = (k2 - k1) / (1 + k1 * k2)
        #判断非数
        if not math.isnan(tan_theta):
            #计算反正切值
            theta = math.atan(tan_theta)
            #将弧度转换为角度
            angle_degrees = math.degrees(theta)

            return angle_degrees
        else:
            #非数
            logger.error("nonumber !")
            return 0
    else:
        #除数为0的情况
        logger.warning("The divisor is 0 !")
        return 0
    
# #两向量计算夹角
# def angle_between_vectors(v1, v2):  
#     dot_product = sum(x * y for x, y in zip(v1, v2))  
#     magnitude_v1 = math.sqrt(sum(x**2 for x in v1))  
#     magnitude_v2 = math.sqrt(sum(x**2 for x in v2))  
#     cos_angle = dot_product / (magnitude_v1 * magnitude_v2)  
#     angle_radians = math.acos(cos_angle)

#     return angle_radians

def angle_between_vectors(a, b):
    # 计算向量的单位向量
    unit_a = a / np.linalg.norm(a)
    unit_b = b / np.linalg.norm(b)
    
    # 计算点积
    dot_product = np.dot(unit_a, unit_b)
    
    # 限制cos值范围，避免数值误差引起的超出[-1, 1]的值
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算夹角（弧度制）
    angle = np.arccos(dot_product)
    
    # 使用叉积判断方向：正为逆时针，负为顺时针
    cross_product = np.cross(a, b)
    if cross_product < 0:
        angle = -angle
    
    return angle
#二维向量更新坐标
def update_vector(present_vector,target_vector):
    rotate_angle = angle_between_vectors(present_vector,target_vector)
    logger.info(f"present_vector: {present_vector} target_vector: {target_vector} Rotate_Angle: {rotate_angle}")

    rotation_matrix = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],  
                                [np.sin(rotate_angle), np.cos(rotate_angle)]])
    update_vector =  np.dot(rotation_matrix, present_vector)

    logger.info(f"update_vector: {update_vector}")

    return update_vector


#骨骼类
class Armature:
    def __init__(self,initialize_pose,ax_initialize):
        self.pose = initialize_pose
        self.ax_initialize = ax_initialize
    def calculate_vector_rotation(self,start_bone,end_bone):
        current = np.array([(self.pose[start_bone][0]-self.pose[end_bone][0]),
                                 (self.pose[start_bone][1]-self.pose[end_bone][1]),
                                 (self.pose[start_bone][2]-self.pose[end_bone][2])])
        target = np.array([(self.target_pose[start_bone][0]-self.target_pose[end_bone][0]),
                                 (self.target_pose[start_bone][1]-self.target_pose[end_bone][1]),
                                 (self.target_pose[start_bone][2]-self.target_pose[end_bone][2])])
        yaw, pitch = angle_between_vectors(current,target)
        logger.info("")
        return yaw,pitch
    def update_pose(self, yaw, pitch,control_bone):
        with open('AMC/bone_paternity.json', 'r') as bone_file:
            bone_paternity = json.load(bone_file)
            bone_file.close()
        #LinkedBones表示该点还关联了哪些点，这个点的运动会让关联点一起运动
        LinkedBones = bone_paternity[control_bone]['LinkedBones']

        # 转换角度为弧度
        theta_yaw_rad = np.radians(yaw)#俯仰轴
        theta_pitch_rad = np.radians(pitch)#航向轴
        
        
        # 创建旋转矩阵
        r_yaw = R.from_euler('x', theta_yaw_rad)   # 绕 X 轴的旋转
        r_pitch = R.from_euler('z', theta_pitch_rad)  # 绕 Z 轴的旋转
        

        if control_bone == 1:
            
            for single_bone in LinkedBones:
                #计算身体自旋转
                k1_current = (self.pose[2][1]-self.pose[1][1])/(self.pose[2][0]-self.pose[1][0])
                k2_target = (self.target_pose[2][1]-self.target_pose[1][1])/(self.target_pose[2][0]-self.target_pose[1][0])
                body_roll = calculate_angle(k1_current,k2_target)

                theta_roll_rad = np.radians(body_roll)#翻滚轴
                r_roll = R.from_euler('y', theta_roll_rad) # 绕 Y 轴的旋转
                # 将控制点转换为相对于当前骨骼的局部坐标
                control_point = self.pose[single_bone]
                local_control_point = control_point - np.array(self.pose[control_bone])
                
                rotated_control_point = r_roll.apply(r_pitch.apply(r_yaw.apply(local_control_point)))
                
                # 将旋转后的控制点转换回全局坐标
                self.pose[single_bone] = rotated_control_point + np.array(self.pose[control_bone])
            write_rotate_angle(1,[pitch,yaw,body_roll])

        else:
            for single_bone in LinkedBones:
                # 将控制点转换为相对于当前骨骼的局部坐标
                control_point = self.pose[single_bone]
                local_control_point = control_point - np.array(self.pose[control_bone])
                
                rotated_control_point = r_pitch.apply(r_yaw.apply(local_control_point))
                
                # 将旋转后的控制点转换回全局坐标
                self.pose[single_bone] = rotated_control_point + np.array(self.pose[control_bone])

if __name__ == "__main__":
    #初始化画布
    init_figure()

    present_vector = np.array([100,-80,-200])
    target_vector = np.array([-60,-200,300])
    logger.info(f"present_vector_3d: {present_vector}, target_vector_3d: {target_vector}")

    initialize_ax_3d.draw_3d_vector(present_vector,"green")
    initialize_ax_3d.draw_3d_vector(target_vector,"black")

    #转为X-Z二维坐标
    logger.info("X-Z Rotate")
    #X-Z转变需要转换x坐标为相反数
    present_vector_xz = np.array([-present_vector[0],present_vector[2]])
    #present_vector_xz = np.array([present_vector[2],present_vector[0]])
    target_vector_xz  = np.array([-target_vector[0],target_vector[2]])
    new_vector_1 = update_vector(present_vector_xz,target_vector_xz)
    #绘制X-Z坐标上的向量
    initialize_xz.draw_2d_vector(present_vector_xz,"green")
    initialize_xz.draw_2d_vector(target_vector_xz,"black")
    initialize_xz.draw_2d_vector(new_vector_1,"red")
    #logger.info(f"new_vector_1: {new_vector_1}")
    
    #转为X-Y二维坐标
    logger.info("X-Y Rotate")
    present_vector_xy = np.array([present_vector[1],-new_vector_1[0]])
    target_vector_xy = np.array([target_vector[1],-target_vector[0]])
    new_vector_2 = update_vector(present_vector_xy,target_vector_xy)
    #绘制X-Z坐标上的向量
    initialize_xy.draw_2d_vector(present_vector_xy,"green")
    initialize_xy.draw_2d_vector(target_vector_xy,"black")
    initialize_xy.draw_2d_vector(new_vector_2,"red")

    #logger.info(f"new_vector_2: {new_vector_2}")

    present_vector = np.array([-new_vector_2[1],new_vector_2[0],new_vector_1[1]])
    
    
    initialize_ax_3d.draw_3d_vector(present_vector,"red")

    #plt.close("all")
    plt.show()
