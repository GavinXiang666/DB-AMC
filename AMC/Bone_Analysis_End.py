
import math
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyfiglet,gc
import json,csv,tqdm
from datetime import datetime
from YoloPose import *
from scipy.spatial.transform import Rotation as R

'''
创建程序日志:
    Bone_Analysis_End.log
'''
logger.add("AMC/log/Bone_Analysis_End.log",rotation="10 MB")
logger.info(f"Start Time: {datetime.now()}") 

#先加载openpose骨骼检测模型,防止冲突
#openpose_detector = None
openpose_detector = Openpose_Detector()
logger.info("Openpose Model is successfully loaded")

#注入灵魂
print(pyfiglet.figlet_format("Dream Busters!",font="slant"))
print(pyfiglet.figlet_format("Bone   Analysis",font="slant"))

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
    global pose_ax_3d
    pose_fig = plt.figure(num="pose figure",figsize=(8, 8))
    pose_ax_3d = Ax(111,pose_fig,'3d')
    pose_ax_3d.set_title('Pose Figure 3D')
    #X-Z画布
    global xz_figure
    xz_figure = plt.figure(num="x-z figure",figsize=(4, 4))
    xz_figure = Ax(111,xz_figure)
    #X-Y画布
    global xy_figure
    xy_figure = plt.figure(num="x-y figure",figsize=(4, 4))
    xy_figure = Ax(111,xy_figure)

#绘制pose
def draw_pose(pose_list,mode="Normal"):
    '''
    mode 绘制模式
    '''
    if mode == "Normal":
        for n in range(0,len(pose_list)):
            for plot_index,plot in enumerate(pose_list[n]):
                if plot_index in [2,3,4,8,9,10]:
                    pose_ax_3d.draw_scatter([plot],color='blue')
                elif plot_index in [5,6,7,11,12,13]:
                    pose_ax_3d.draw_scatter([plot],color='red')
                else:
                    pose_ax_3d.draw_scatter([plot],color='green')
    elif mode == "compare":
        pass
    
    
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

#计算2d,3d空间两点间的距离
def calculate_distance(point1,point2,mode="3D"):
    if mode == "3D":
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
    elif mode == "2D":
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
#计算2d,3d空间中两点的中点
def calculate_middle_point(point1,point2,mode="3D"):
    if mode == "3D":
        return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2]
    elif mode == "2D":
        return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

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


'''
视频处理,骨骼分析类
'''
class Dream_Eyes:
    def __init__(self,front_video,left_video,right_video):
        self.front_video = front_video
        self.left_video = left_video
        self.right_video = right_video
        self.eye = openpose_detector
        #载入待分析的视频
        try:
            self.front_detector = cv2.VideoCapture(self.front_video)
            self.left_detector = cv2.VideoCapture(self.left_video)
            self.right_detector = cv2.VideoCapture(self.right_video)
            logger.info("Video is successfully loaded")
        except:
            logger.error("Video is failed loaded")
        logger.info(f"front video frames: {self.front_detector.get(cv2.CAP_PROP_FRAME_COUNT)} left video frames:{self.left_detector.get(cv2.CAP_PROP_FRAME_COUNT)} right video frames: {self.right_detector.get(cv2.CAP_PROP_FRAME_COUNT)}")
        #获取最小的那个视频的总帧数,防止获取时越界
        self.total_frame_count = min(self.front_detector.get(cv2.CAP_PROP_FRAME_COUNT),
                                    self.left_detector.get(cv2.CAP_PROP_FRAME_COUNT),self.right_detector.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"video frames: {self.total_frame_count}")
    def view(self,frame_index):
        view_result = []
        detector_sets=[
            self.front_detector,
            self.left_detector,
            self.right_detector
        ]
        for cap_index,cap in enumerate(detector_sets):
            #设置视频到指定帧数
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            #读取帧数
            ret, particular_frame = cap.read()
            #若读取成功
            if ret == True:
                #添加当前帧
                view_result.append(particular_frame)
            else:
                logger.error(f"View Failed.--->frame_index: {frame_index} cap_index: {cap_index}")
        
        return view_result
    '''
    骨骼识别
    '''
    def detect_pose(self,view):
        pose_info = []
        for view_index,view_image in enumerate(view):
            #[0]为图像,[1]为骨骼识别信息
            pose_data = self.eye.detect_pose(view_image)[1]
            logger.info(f'index: {view_index}  pose_info: {pose_data}')
            #添加信息
            pose_info.append(pose_data)
            
        return pose_info
    '''
    整合骨骼信息
    '''
    def intergrate_pose_info(self):
        json_data = read_pose_data(r'AMC\output\pose_data.json')
        intergrated_pose = []
        intergrated_pose_list = []
        for pose_index in range(0,len(json_data)):
        #for pose_index in range(0,3):
            pose_data = json_data[pose_index]['pose_info']
            '''
            pose_data[n]--->n为摄像头id,0-front,1-left,2-right
            '''
            #print(pose_index,pose_data[0],pose_data[1],pose_data[2])
            #X-Z
            for camera_id in range(0,3):
                intergrated_pose = [[-sublist[0], 0, -sublist[1]] for sublist in pose_data[0]]
                print("inter",intergrated_pose)
                try:
                    for plot_index,plot in enumerate(pose_data[camera_id]):
                        if plot_index in [2,3,4,8,9,10]:
                            intergrated_pose[plot_index][1] = pose_data[2][plot_index][0]
                            #intergrated_pose[plot_index].append(pose_data[2][plot_index][0])
                            
                        elif plot_index in [5,6,7,11,12,13]:
                            intergrated_pose[plot_index][1] = pose_data[1][plot_index][0]
                            #intergrated_pose[plot_index].append(pose_data[1][plot_index][0])
                            
                        else:
                            intergrated_pose[plot_index][1] = pose_data[2][plot_index][0]
                            #intergrated_pose[plot_index].append(pose_data[2][plot_index][0])
                    pose_middle_point = calculate_middle_point(point1=intergrated_pose[8],point2=intergrated_pose[11],mode="3D")
                    #计算与原点(0,0,0)距离
                    delta_x = pose_middle_point[0] - 0
                    delta_y = pose_middle_point[1] - 0
                    delta_z = pose_middle_point[2] - 0
                    for new_plot_index,plot in enumerate(intergrated_pose):
                        intergrated_pose[new_plot_index][0] = plot[0] - delta_x
                        intergrated_pose[new_plot_index][1] = plot[1] - delta_y
                        intergrated_pose[new_plot_index][2] = plot[2] - delta_z
                    intergrated_pose_list.append(intergrated_pose)

                except:
                    
                    intergrated_pose_list.append(intergrated_pose)

        return intergrated_pose_list
#二维向量更新坐标
def update_vector(present_vector,target_vector):
    rotate_angle = angle_between_vectors(present_vector,target_vector)
    logger.info(f"present_vector: {present_vector} target_vector: {target_vector} Rotate_Angle: {rotate_angle}")

    rotation_matrix = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],  
                                [np.sin(rotate_angle), np.cos(rotate_angle)]])
    update_vector =  np.dot(rotation_matrix, present_vector)

    logger.info(f"update_vector: {update_vector}")

    return update_vector,rotate_angle
'''
骨骼类
'''
class Armature: 
    def __init__(self,pose):
        self.pose = pose
        print('Armature: ',self.pose)
    #移动坐标轴
    def update_position(self,vector,old_origin_point,new_origin_point):
        new_origin_x = new_origin_point[0]
        new_origin_y = new_origin_point[1]
        new_origin_z = new_origin_point[2]
        changed_vectors = []
        for change_vector in vector:
            delta_x = new_origin_x - old_origin_point[0]
            delta_y = new_origin_y - old_origin_point[1]
            delta_z = new_origin_z - old_origin_point[2]
            change_vector = np.array([change_vector[0]-delta_x,change_vector[1]-delta_y,change_vector[2]-delta_z])
            
            changed_vectors.append(change_vector)
        
        return changed_vectors
    def calculate_vector_rotation(self,start_bone,end_bone):
        present_vector = np.array([(self.pose[start_bone][0]-self.pose[end_bone][0]),
                                 (self.pose[start_bone][1]-self.pose[end_bone][1]),
                                 (self.pose[start_bone][2]-self.pose[end_bone][2])])
        target_vector = np.array([(self.target_pose[start_bone][0]-self.target_pose[end_bone][0]),
                                 (self.target_pose[start_bone][1]-self.target_pose[end_bone][1]),
                                 (self.target_pose[start_bone][2]-self.target_pose[end_bone][2])])
        logger.info(f'present vector:{present_vector} target vector {target_vector}')

        #X-Z转变需要转换x坐标为相反数
        present_vector_xz = np.array([-present_vector[0],present_vector[2]])
        target_vector_xz  = np.array([-target_vector[0],target_vector[2]])
        inf_vector = update_vector(present_vector_xz,target_vector_xz)

        new_vector_1 = inf_vector[0]
        yaw = inf_vector[1]
        #转为X-Y二维坐标
        present_vector_xy = np.array([present_vector[1],-new_vector_1[0]])
        target_vector_xy = np.array([target_vector[1],-target_vector[0]])
        inf_vector = update_vector(present_vector_xy,target_vector_xy)

        new_vector_2 = inf_vector[0]
        pitch = inf_vector[1]

        new_present_vector = np.array([-new_vector_2[1],new_vector_2[0],new_vector_1[1]])
        print(new_present_vector)

        return yaw,pitch
    def update_pose(self,yaw, pitch,control_bone):
        with open('AMC/bone_paternity.json', 'r') as bone_file: 
            bone_paternity = json.load(bone_file)
            bone_file.close()
        print(bone_paternity)
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
                body_roll = 0
            write_rotate_angle(1,[pitch,yaw,body_roll])

        else:
            for single_bone in LinkedBones:
                #将控制点转换为相对于当前骨骼的局部坐标
                control_point = self.pose[single_bone]
                local_control_point = control_point - np.array(self.pose[control_bone])
                
                rotated_control_point = r_pitch.apply(r_yaw.apply(local_control_point))

                #将旋转后的控制点转换回全局坐标
                self.pose[single_bone] = rotated_control_point + np.array(self.pose[control_bone])

    def rotate(self,target_pose):
        #获取目标骨骼信息
        self.target_pose = target_pose
        """
        旋转身体部分Body
        """
        logger.info("Body Rotation")
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
        '''
        身体旋转计算单独写
        '''
        #X-Z转变需要转换x坐标为相反数
        present_vector_xz = np.array([-body_current[0],body_current[2]])
        target_vector_xz  = np.array([-body_target[0],body_target[2]])
        inf_vector = update_vector(present_vector_xz,target_vector_xz)
        new_vector_1 = inf_vector[0]
        body_xz_rotation = inf_vector[1]
        #转为X-Y二维坐标
        present_vector_xy = np.array([body_current[1],-new_vector_1[0]])
        target_vector_xy = np.array([body_target[1],-body_target[0]])
        
        inf_vector = update_vector(present_vector_xy,target_vector_xy)
        new_vector_2 = inf_vector[0]
        body_xy_rotation = inf_vector[1]
        new_body_vector = np.array([-new_vector_2[1],new_vector_2[0],new_vector_1[1]])

        self.update_pose(body_xz_rotation, body_xy_rotation,control_bone=1)

        """
        旋转头部Head
        """
        logger.info("Head Rotation")
        #移动坐标轴
        head_origin_point = self.pose[1]
        self.update_position(vector=self.pose,old_origin_point=np.array([0,0,0]),new_origin_point=head_origin_point)
        
        yaw, pitch = self.calculate_vector_rotation(1,0)
        self.update_pose(yaw, pitch,control_bone=0)
        write_rotate_angle(0,[pitch,yaw])

        """
        旋转左大手臂Arm_L
        """
        logger.info("Arm_L Rotation")
        #移动坐标轴
        Arm_L_origin_point = self.pose[2]
        self.update_position(vector=self.pose,old_origin_point=head_origin_point,new_origin_point=Arm_L_origin_point)
        
        yaw, pitch = self.calculate_vector_rotation(2,3)
        self.update_pose(yaw, pitch,control_bone=2)
        write_rotate_angle(2,[pitch,yaw])
        """
        旋转左小臂Forearm_L
        """
        logger.info("Forearm_L Rotation")
        #移动坐标轴
        Forearm_L_origin_point = self.pose[3]
        self.update_position(vector=self.pose,old_origin_point=Arm_L_origin_point,new_origin_point=Forearm_L_origin_point)
        
        yaw, pitch = self.calculate_vector_rotation(3,4)
        self.update_pose(yaw, pitch,control_bone=3)
        write_rotate_angle(3,[pitch,yaw])

        """
        旋转右大手臂Arm_R
        """
        logger.info("Arm_R Rotation")
        #移动坐标轴
        Arm_R_origin_point = self.pose[5]
        self.update_position(vector=self.pose,old_origin_point=Forearm_L_origin_point,new_origin_point=Arm_R_origin_point)
        
        yaw, pitch = self.calculate_vector_rotation(5,6)
        self.update_pose(yaw, pitch,control_bone=5)
        write_rotate_angle(5,[pitch,yaw])

        """
        旋转右小臂Forearm_R
        """
        logger.info("Forearm_R Rotation")
        #移动坐标轴
        Forearm_R_origin_point = self.pose[6]
        self.update_position(vector=self.pose,old_origin_point=Arm_R_origin_point,new_origin_point=Forearm_R_origin_point)
        
        yaw, pitch = self.calculate_vector_rotation(6,7)
        self.update_pose(yaw, pitch,control_bone=6)
        write_rotate_angle(6,[pitch,yaw])

        """
        旋转左大腿Thigh_L
        """
        logger.info("Thigh_L Rotation")
        #移动坐标轴
        Thigh_L_origin_point = self.pose[9]
        self.update_position(vector=self.pose,old_origin_point=Forearm_R_origin_point,new_origin_point=Thigh_L_origin_point)

        yaw, pitch = self.calculate_vector_rotation(9,8)
        self.update_pose(yaw, pitch,control_bone=8)
        write_rotate_angle(8,[pitch,yaw])
        """
        旋转左小腿Shin_L
        """
        logger.info("Shin_L Rotation")
        #移动坐标轴
        Shin_L_origin_point = self.pose[10]
        self.update_position(vector=self.pose,old_origin_point=Thigh_L_origin_point,new_origin_point=Shin_L_origin_point)

        yaw, pitch = self.calculate_vector_rotation(10,9)
        self.update_pose(yaw, pitch,control_bone=9)
        write_rotate_angle(9,[pitch,yaw])

        """
        旋转右大腿Thigh_R
        """
        logger.info("Thigh_R Rotation")
        #移动坐标轴
        Thigh_R_origin_point = self.pose[11]
        self.update_position(vector=self.pose,old_origin_point=Shin_L_origin_point,new_origin_point=Thigh_R_origin_point)

        yaw, pitch = self.calculate_vector_rotation(12,11)
        self.update_pose(yaw, pitch,control_bone=11)
        write_rotate_angle(11,[pitch,yaw])

        """
        旋转右小腿Shin_R
        """
        logger.info("Shin_R Rotation")
        #移动坐标轴
        Shin_R_origin_point = self.pose[12]
        self.update_position(vector=self.pose,old_origin_point=Thigh_R_origin_point,new_origin_point=Shin_R_origin_point)

        yaw, pitch = self.calculate_vector_rotation(13,12)
        self.update_pose(yaw, pitch,control_bone=12)
        write_rotate_angle(12,[pitch,yaw])

if __name__ == "__main__":      
    #抽帧常数
    frame_extraction = 8
    #DreamBusters Eyes
    Dream_Eye = Dream_Eyes(front_video=r"AMC\output\video\New New Camera 2.avi",
                            left_video=r"AMC\output\video\New New Camera 3.avi",right_video=r"AMC\output\video\New New Camera 0.avi")
    
    all_pose_data = []
    #根据抽帧常数抽帧
    for frame_index in range(100,150):
    #for frame_index in range(0,int(Dream_Eye.total_frame_count // frame_extraction)):
        view_result = Dream_Eye.view(frame_index=frame_index)
        pose_result = Dream_Eye.detect_pose(view_result)
        data_pose = {
            "frame_index":frame_index,  
            "pose_info":pose_result
        }
        all_pose_data.append(data_pose)
    
    with open(r'AMC\output\pose_data.json', 'w', encoding='utf-8') as pose_file:
        json.dump(all_pose_data, pose_file, ensure_ascii=False, indent=4)

    logger.info("============ pose_data.json写入完成 ============")

    #整合骨骼信息
    intergrated_pose_list = Dream_Eye.intergrate_pose_info()
    logger.info("============ 骨骼信息整合完成 ============")

    #根据整合信息创建骨骼对象
    armature_object_list = []
    for armature in intergrated_pose_list:
        new_armature = Armature(armature)
        #添加对象
        armature_object_list.append(new_armature)
    logger.info("============ 骨骼对象创建完成 ============")

    # print(armature_object_list[0].pose)
    # armature_object_list[1].rotate(armature_object_list[0].pose)
    logger.info("============ 开始计算骨骼旋转 ============")
    for i,armature_object in enumerate(armature_object_list):
        if i >= 1:
            armature_object_list[i].rotate(armature_object_list[i-1].pose)
        else:
            continue
    # #初始化画布
    init_figure()
    logger.info("============ 画布初始化完成 ============")

    draw_pose(intergrated_pose_list,mode="Normal")

    logger.info("*********** 可视化开始 ***********")
    #plt.show()
