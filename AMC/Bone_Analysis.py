
from camera import *
from YoloPose import *
import matplotlib,math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation as R
import json,csv,os,sys
import pyfiglet,gc

#注入灵魂
print(pyfiglet.figlet_format("Dream Busters!",font="slant"))
print(pyfiglet.figlet_format("Bone   Analysis",font="slant"))
#先加载模型,防止冲突
yolo_detector = Yolo_Detector("cuda")
openpose_detector = Openpose_Detector()

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

def safe_divide(delta_y,delta_x):
    k = delta_y / delta_x
    #判断斜率是否为无效值nan(Not a number)
    if not math.isnan(k):
        return k
    else:
        #print("ZeroDivisionError")
        return 0
#计算两直线夹角
def calculate_angle(k1, k2):
    tan_theta = (k2 - k1) / (1 + k1 * k2)
    if not math.isnan(tan_theta):
        #计算反正切值
        theta = math.atan(tan_theta)
        #将弧度转换为角度
        angle_degrees = math.degrees(theta)

        return angle_degrees
    else:
        return 0
    
"""
Rotate
"""
# def calculate_yaw_pitch_for_segments(current, target):
#     """计算线段 a 和 b 的航向角、俯仰角"""
#     # 归一化向量
#     a_norm = current / np.linalg.norm(current)
#     b_norm = target / np.linalg.norm(target)
    
#     # 计算航向角（Z 轴旋转）
#     yaw_a = np.arctan2(a_norm[1], a_norm[0])
#     yaw_b = np.arctan2(b_norm[1], b_norm[0])
#     theta_yaw = yaw_b - yaw_a

#     # 计算俯仰角（Y 轴旋转）
#     pitch_a = np.arctan2(a_norm[2], np.sqrt(a_norm[0]**2 + a_norm[1]**2))
#     pitch_b = np.arctan2(b_norm[2], np.sqrt(b_norm[0]**2 + b_norm[1]**2))
#     theta_pitch = pitch_b - pitch_a
    
#     return np.degrees(theta_yaw), np.degrees(theta_pitch)
def calculate_yaw_pitch_for_segments(current, target):
    """计算线段 a 和 b 的航向角、俯仰角"""
    # 归一化向量
    a_norm = current / np.linalg.norm(current)
    b_norm = target / np.linalg.norm(target)
    
    # 计算两个向量之间的旋转轴
    cross_product = np.cross(a_norm, b_norm)
    cross_product_norm = np.linalg.norm(cross_product)

    if cross_product_norm == 0:  # 如果两个向量平行
        return 0, 0

    # 计算两个向量之间的旋转角度
    dot_product = np.dot(a_norm, b_norm)
    theta_total = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 总的旋转角度

    # 计算航向角（绕 Z 轴）
    theta_yaw = np.arctan2(b_norm[1], b_norm[0]) - np.arctan2(a_norm[1], a_norm[0])

    # 计算俯仰角（绕 Y 轴）
    theta_pitch = np.arctan2(b_norm[2], np.sqrt(b_norm[0]**2 + b_norm[1]**2)) - \
                  np.arctan2(a_norm[2], np.sqrt(a_norm[0]**2 + a_norm[1]**2))
    
    return np.degrees(theta_yaw), np.degrees(theta_pitch)



class Ax:
    def __init__(self,layout,fig,projection=None):
        self.projection = projection
        self.layout = layout
        self.fig = fig
        self.ax = self.fig.add_subplot(self.layout, projection=self.projection)
        self.ax.set_xlim(-400, 400)  
        self.ax.set_ylim(-600, 600)  
        #设置坐标系与OpenCV一致
        if self.projection is None:
            self.ax.invert_yaxis()
        else:
            self.ax.set_zlim(-400, 400)
            self.ax.set_xlabel('X')  
            self.ax.set_ylabel('Y')  
            self.ax.set_zlabel('Z')
            self.ax.view_init(elev=-90, azim=-90,roll=0)

    def set_title(self,title_name):
        self.ax.set_title(title_name)

    def draw_scatter(self,scatter_list,color):
        for scatter in scatter_list:
            try:
                x, y, z = scatter
                self.ax.scatter(x,y,z,s=2,color=color)
            except:
                x, y = scatter
                self.ax.scatter(x,y,s=2,color=color)

#Magic,don't touch
matplotlib.pyplot.switch_backend('Agg')  # 切换后端以确保与当前环境兼容
matplotlib.pyplot.switch_backend('TkAgg')  # 切换回默认后端

#3D plot(initialize)
initialize_fig = plt.figure(num="initialize pose",figsize=(12, 4))
initialize_ax_3d = Ax(121,initialize_fig,'3d')
initialize_ax_3d.set_title('3D Composite Scene')
target_ax_3d = Ax(122,initialize_fig,'3d')

current_fig = plt.figure(num="current pose",figsize=(5, 5))
current_ax_3d = Ax(111,current_fig,'3d')
current_ax_3d.set_title('Current Pose')
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
        yaw, pitch = calculate_yaw_pitch_for_segments(current,target)

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
            

    def rotate(self,target_pose):
        #获取目标骨骼信息
        self.target_pose = target_pose
        #创建新的骨骼点位置
        #self.pose += [[0,0,0] for _ in range(4)]
        #self.target_pose += [[0,0,0] for _ in range(4)]
        """
        平移当前姿态至目标姿态原点
        """
        x_move = self.target_pose[1][0] - self.pose[1][0]
        y_move = self.target_pose[1][1] - self.pose[1][1]
        z_move = self.target_pose[1][2] - self.pose[1][2]
        #进行整体平移
        for i in range(len(self.pose)):
            self.pose[i][0] = self.pose[i][0] + x_move
            self.pose[i][1] = self.pose[i][1] + y_move
            self.pose[i][2] = self.pose[i][2] + z_move
            self
        """
        旋转身体部分
        """
        #骨骼点8,11连线的中点，用于连接1来计算身体部分旋转
        current_scatter_x = (self.pose[8][0] + self.pose[11][0]) / 2
        current_scatter_y = (self.pose[8][1] + self.pose[11][1]) / 2
        current_scatter_z = (self.pose[8][2] + self.pose[11][2]) / 2
        new_scatter_x = (self.target_pose[8][0] + self.target_pose[11][0]) / 2
        new_scatter_y = (self.target_pose[8][1] + self.target_pose[11][1]) / 2
        new_scatter_z = (self.target_pose[8][2] + self.target_pose[11][2]) / 2

        #初始化姿态身体向量
        body_current = np.array([current_scatter_x-self.pose[1][0],
                                 current_scatter_y-self.pose[1][1],current_scatter_z-self.pose[1][2]])
        #目标姿态身体向量
        body_target = np.array([new_scatter_x-self.target_pose[1][0],
                                new_scatter_y-self.target_pose[1][1],new_scatter_z-self.target_pose[1][2]])

        yaw, pitch = calculate_yaw_pitch_for_segments(body_current,body_target)
        self.update_pose(yaw, pitch,control_bone=1)
        
        """
        旋转头部
        """
        yaw, pitch = self.calculate_vector_rotation(1,0)
        self.update_pose(yaw, pitch,control_bone=0)
        write_rotate_angle(0,[pitch,yaw])
        """
        旋转左大臂,右大臂
        """
        yaw, pitch = self.calculate_vector_rotation(2,3)
        self.update_pose(yaw, pitch,control_bone=2)
        write_rotate_angle(2,[pitch,yaw])
        yaw, pitch = self.calculate_vector_rotation(5,6)
        self.update_pose(yaw, pitch,control_bone=5)
        write_rotate_angle(5,[pitch,yaw])
        """
        旋转左小臂,右小臂
        """
        yaw, pitch = self.calculate_vector_rotation(3,4)
        self.update_pose(yaw, pitch,control_bone=3)
        write_rotate_angle(3,[pitch,yaw])
        yaw, pitch = self.calculate_vector_rotation(6,7)
        self.update_pose(yaw, pitch,control_bone=6)
        write_rotate_angle(6,[pitch,yaw])
        """
        旋转左大腿,右大腿
        """
        #先创建20,21骨骼点用于旋转8,11
        # self.pose[20][0] = self.pose[8][0];self.pose[21][0] = self.pose[11][0]
        # self.pose[20][1] = self.pose[8][1] + 1;self.pose[21][1] = self.pose[11][1] + 1
        # self.pose[20][2] = self.pose[8][2];self.pose[21][2] = self.pose[11][2]

        # self.target_pose[20][0] = self.target_pose[8][0];self.target_pose[21][0] = self.target_pose[11][0]
        # self.target_pose[20][1] = self.target_pose[8][1] + 1;self.target_pose[21][1] = self.target_pose[11][1] + 1
        # self.target_pose[20][2] = self.target_pose[8][2];self.target_pose[21][2] = self.target_pose[11][2]
        # yaw, pitch = self.calculate_vector_rotation(20,8)
        # self.update_pose(yaw, pitch,control_bone=20)
        # yaw, pitch = self.calculate_vector_rotation(21,11)
        # self.update_pose(yaw, pitch,control_bone=21)

        yaw, pitch = self.calculate_vector_rotation(9,8)
        self.update_pose(yaw, pitch,control_bone=8)
        write_rotate_angle(8,[pitch,yaw])
        yaw, pitch = self.calculate_vector_rotation(12,11)
        self.update_pose(yaw, pitch,control_bone=11)
        write_rotate_angle(11,[pitch,yaw])
        """
        旋转左小腿,右小腿
        """
        yaw, pitch = self.calculate_vector_rotation(9,10)
        self.update_pose(yaw, pitch,control_bone=9)
        write_rotate_angle(9,[pitch,yaw])
        yaw, pitch = self.calculate_vector_rotation(13,12)
        self.update_pose(yaw, pitch,control_bone=12)
        write_rotate_angle(12,[pitch,yaw])
        

if __name__ == "__main__":
    """
    初始化动作
    """
    front = cv2.imread("AMC/output/image/front.png")
    left = cv2.imread("AMC/output/image/left.png")
    right = cv2.imread("AMC/output/image/right.png")

    camera_frame = [cv2.imread("AMC/output/image/front.png")]

    with open('AMC/pose_data.json', 'w') as f:  
        pass
    pose_file = open('AMC/pose_data.json', 'w')

    pose_data_list = []
    for i,image in enumerate(camera_frame):
        initialize_info = openpose_detector.detect_pose(image)
        initialize_pose_info = initialize_info[1]
        print(f'initialize_pose_info: {initialize_pose_info}')

        position = "front" if i == 0 else ("left" if i == 1 else "right") 
        pose_data = {
            position:[
                {"initialize_pose_info":initialize_pose_info}
            ]
        }
        pose_data_list.append(pose_data)
    #写入初始pose数据
    json.dump(pose_data_list, pose_file,indent=4)

    pose_file.close()

    initialize_pose_data = read_pose_data('AMC/output/pose_data.json')
    initialize_front_plot_list = initialize_pose_data[0]['front'][0]['initialize_pose_info']
    initialize_left_plot_list = initialize_pose_data[1]['left'][0]['initialize_pose_info']
    initialize_right_plot_list = initialize_pose_data[2]['right'][0]['initialize_pose_info']
 
    for plot_index,plot in enumerate(initialize_front_plot_list):
        if plot_index in [2,3,4,8,9,10]:
            initialize_front_plot_list[plot_index].append(initialize_right_plot_list[plot_index][0])
            
        elif plot_index in [5,6,7,11,12,13]:
            initialize_front_plot_list[plot_index].append(initialize_left_plot_list[plot_index][0])
            
        else:
            initialize_front_plot_list[plot_index].append(initialize_right_plot_list[plot_index][0])
            
    for plot_index,plot in enumerate(initialize_front_plot_list):
        if plot_index in [2,3,4,8,9,10]:
            initialize_ax_3d.draw_scatter([plot],color='blue')
        elif plot_index in [5,6,7,11,12,13]:
            initialize_ax_3d.draw_scatter([plot],color='red')
        else:
            initialize_ax_3d.draw_scatter([plot],color='green')
    initialize_pose_plot_list = initialize_front_plot_list
    """
    下一帧动作的骨骼位置
    """
    front_image = cv2.imread("AMC/output/image/image_front.jpg")
    left_image = cv2.imread("AMC/output/image/image_left.jpg")
    right_image = cv2.imread("AMC/output/image/image_right.jpg")
    camera_image = [front_image,left_image,right_image]
    
    next_pose_info_list = []
    for image in camera_image:
        #Yolov5+Openpose检测
        yolo_image = yolo_detector.detect_person(image)
        next_info = openpose_detector.detect_pose(yolo_image)
        next_pose_info = next_info[1]
        next_pose_info_list.append(next_pose_info)

    target_pose_plot_list =  [list(item) for item in next_pose_info_list[0]]
    


    """
    合成下一帧骨骼动作3d信息
    """
    for plot_index,plot in enumerate(target_pose_plot_list):
        if plot_index in [2,3,4,8,9,10]:
            target_pose_plot_list[plot_index].append(next_pose_info_list[1][plot_index][0])
            
        elif plot_index in [5,6,7,11,12,13]:
            target_pose_plot_list[plot_index].append(next_pose_info_list[2][plot_index][0])
            
        else:
            target_pose_plot_list[plot_index].append(next_pose_info_list[2][plot_index][0])
    
    #绘制目标骨骼图像
    for plot_index,target_plot in enumerate(target_pose_plot_list):
        if plot_index in [2,3,4,8,9,10]:
            target_ax_3d.draw_scatter([target_plot],color='blue')
        elif plot_index in [5,6,7,11,12,13]:
            target_ax_3d.draw_scatter([target_plot],color='red')
        else:
            target_ax_3d.draw_scatter([target_plot],color='green')
    
    pose = Armature(initialize_pose_plot_list,target_ax_3d)
    
    pose.rotate(target_pose=target_pose_plot_list)
    #绘制旋转后的姿态
    for plot_index,plot in enumerate(pose.pose):
        if plot_index in [2,3,4,8,9,10]:
            current_ax_3d.draw_scatter([plot],color='blue')
        elif plot_index in [5,6,7,11,12,13]:
            current_ax_3d.draw_scatter([plot],color='red')
        else:
            current_ax_3d.draw_scatter([plot],color='green')

    plt.show(block=True)

    