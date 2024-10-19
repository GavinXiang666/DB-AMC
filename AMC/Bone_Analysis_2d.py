
from camera import *
from YoloPose import *
import matplotlib,math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import json,csv,os,sys
import pyfiglet,gc

#注入灵魂
print(pyfiglet.figlet_format("Dream Busters!",font="slant"))
print(pyfiglet.figlet_format("Bone   Analysis",font="slant"))
#先加载模型,防止冲突
yolo_detector = Yolo_Detector("cpu")
openpose_detector = Openpose_Detector()

#创建骨骼旋转文件
with open('AMC/output/bone_rotate_angle.csv', 'w', newline='') as angle_file:
    csv_writer = csv.writer(angle_file)
    #写入标题行
    csv_writer.writerow(["position", "bone_id", "rotate_angle"])
    #angle_file.close()

def write_rotate_angle(position,bone_id,rotate_angle):
    #追加模式(a),防止覆盖
    with open('AMC/output/bone_rotate_angle.csv', 'a', newline='') as angle_file:
        csv_writer = csv.writer(angle_file)
        csv_writer.writerow([position,bone_id,rotate_angle])

def read_pose_data(file_path):
    with open(file_path, 'r') as pose_data_json:
        pose_data = json.load(pose_data_json)
        pose_data_json.close()
    return pose_data


    
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
            except:
                x, y = scatter
            self.ax.scatter(x,y,s=2,color=color)

#Magic,don't touch
matplotlib.pyplot.switch_backend('Agg')  # 切换后端以确保与当前环境兼容
matplotlib.pyplot.switch_backend('TkAgg')  # 切换回默认后端

#3D plot
initialize_fig = plt.figure(num="initialize pose",figsize=(8, 8))
initialize_ax_3d = Ax(221,initialize_fig,'3d')
initialize_ax_3d.set_title('3D Composite Scene')
initialize_ax_front = Ax(222,initialize_fig)
initialize_ax_front.set_title('front_camera')
initialize_ax_left = Ax(223,initialize_fig)
initialize_ax_left.set_title('left_camera')
initialize_ax_right = Ax(224,initialize_fig)
initialize_ax_right.set_title('right_camera')
#front camera
simulate_fig_front = plt.figure(num="Bone movement simulation(front)",figsize=(6.5, 2))
front_ax_initialize = Ax(141,simulate_fig_front)
front_ax_initialize.set_title("initialize")
front_ax_step1 = Ax(142,simulate_fig_front)
front_ax_step1.set_title("step1")
front_ax_step2 = Ax(143,simulate_fig_front)
front_ax_step2.set_title("step2")
front_ax_step3 = Ax(144,simulate_fig_front)
front_ax_step3.set_title("step3")
#left camera
simulate_fig_left = plt.figure(num="Bone movement simulation(left)",figsize=(6.5, 2))
left_ax_initialize = Ax(141,simulate_fig_left)
left_ax_initialize.set_title("initialize")
left_ax_step1 = Ax(142,simulate_fig_left)
left_ax_step1.set_title("step1")
left_ax_step2 = Ax(143,simulate_fig_left)
left_ax_step2.set_title("step2")
left_ax_step3 = Ax(144,simulate_fig_left)
left_ax_step3.set_title("step3")
#right camera
simulate_fig_right = plt.figure(num="Bone movement simulation(right)",figsize=(6.5, 2))
right_ax_initialize = Ax(141,simulate_fig_right)
right_ax_initialize.set_title("initialize")
right_ax_step1 = Ax(142,simulate_fig_right)
right_ax_step1.set_title("step1")
right_ax_step2 = Ax(143,simulate_fig_right)
right_ax_step2.set_title("step2")
right_ax_step3 = Ax(144,simulate_fig_right)
right_ax_step3.set_title("step3")

#骨骼类
class Armature:
    def __init__(self,initialize_pose,ax_initialize,ax_step1,ax_step2,ax_step3,camera_position):
        self.pose = initialize_pose
        self.ax_initialize = ax_initialize
        self.ax_step1 = ax_step1
        self.ax_step2 = ax_step2
        self.ax_step3 = ax_step3
        self.camera_position = camera_position
    #模拟骨骼旋转的坐标变换
    def update_pose(self,rotate_angle,control_bone):
        with open('AMC/bone_paternity.json', 'r') as bone_file:
            bone_paternity = json.load(bone_file)
            bone_file.close()
        LinkedBones = bone_paternity[control_bone]['LinkedBones']
        #print(f"control bone {control_bone} ---> ",LinkedBones)
        #print('self.pose',self.pose,'len:',len(self.pose))
        origin_x,origin_y = self.pose[control_bone]

        #每个从动点以主点为中心旋转，可看作点在圆上的运动
        for single_bone in LinkedBones:
            driven_point_x,driven_point_y = self.pose[single_bone]
            
            #计算从动点到主点的相对位置,用于方便重新建系
            relative_x = driven_point_x - origin_x
            relative_y = driven_point_y - origin_y
            
            # 将角度转换为弧度
            rotate_angle = math.radians(rotate_angle)
            # 使用旋转矩阵计算新坐标
            new_x = origin_x + (relative_x * math.cos(rotate_angle) - relative_y * math.sin(rotate_angle))
            new_y = origin_y + (relative_x * math.sin(rotate_angle) + relative_y * math.cos(rotate_angle))

            #更新self.pose信息
            self.pose[single_bone][0] = new_x
            self.pose[single_bone][1] = new_y

    def rotate(self,bone):

        self.target_bone = bone
        
        '''
        旋转身体部分
        '''
        #骨骼点8,11连线的中点，用于连接1来计算身体部分旋转
        current_scatter_x = (self.pose[8][0] + self.pose[11][0]) / 2
        current_scatter_y = (self.pose[8][1] + self.pose[11][1]) / 2
        new_scatter_x = (self.target_bone[8][0] + self.target_bone[11][0]) / 2
        new_scatter_y = (self.target_bone[8][1] + self.target_bone[11][1]) / 2
        k2_target = safe_divide((new_scatter_y - self.target_bone[1][1]) , (new_scatter_x - self.target_bone[1][0]))
        k1_current = safe_divide((current_scatter_y - self.pose[1][1]) , (current_scatter_x - self.pose[1][0]))

        body_angle = calculate_angle(k1_current,k2_target)
        #print(f"body angle: {body_angle}")

        #更新旋转后的骨骼点坐标
        self.update_pose(body_angle,control_bone=1)
        write_rotate_angle(self.camera_position,'1',body_angle)

        '''
        旋转肩膀,头部
        '''
        #左肩
        k1_current = safe_divide((self.pose[1][1] - self.pose[2][1]) , (self.pose[1][0] - self.pose[2][0]))
        k2_target = safe_divide((self.target_bone[1][1] - self.target_bone[2][1]) , (self.target_bone[1][0] - self.target_bone[2][0]))
        left_shoulder_angle = calculate_angle(k1_current, k2_target)
        self.update_pose(left_shoulder_angle,control_bone=18)
        write_rotate_angle(self.camera_position,'18',left_shoulder_angle)

        #右肩
        k1_current = safe_divide((self.pose[1][1] - self.pose[5][1]) , (self.pose[1][0] - self.pose[5][0]))
        k2_target = safe_divide((self.target_bone[1][1] - self.target_bone[5][1]) , (self.target_bone[1][0] - self.target_bone[5][0]))
        right_shoulder_angle = calculate_angle(k1_current, k2_target)
        self.update_pose(right_shoulder_angle,control_bone=19)
        write_rotate_angle(self.camera_position,'19',right_shoulder_angle)

        #头部
        k1_current = safe_divide((self.pose[1][1] - self.pose[0][1]) , (self.pose[1][0] - self.pose[0][0]))
        k2_target = safe_divide((self.target_bone[1][1] - self.target_bone[0][1]) , (self.target_bone[1][0] - self.target_bone[0][0]))
        head_angle = calculate_angle(k1_current, k2_target)
        self.update_pose(head_angle,control_bone=0)
        write_rotate_angle(self.camera_position,'0',head_angle)

        self.ax_step1.draw_scatter(self.pose,color = 'g')
        '''
        旋转大臂,大腿
        '''
        
        #左大臂
        k1_current = safe_divide((self.pose[2][1] - self.pose[3][1]) , (self.pose[2][0] - self.pose[3][0]))
        k2_target = safe_divide((self.target_bone[2][1] - self.target_bone[3][1]) , (self.target_bone[2][0] - self.target_bone[3][0]))
        left_upper_arm_angle = calculate_angle(k1_current,k2_target)
        self.update_pose(left_upper_arm_angle,control_bone=2)
        write_rotate_angle(self.camera_position,'2',left_upper_arm_angle)

        #右大臂
        k1_current = safe_divide((self.pose[5][1] - self.pose[6][1]) , (self.pose[5][0] - self.pose[6][0]))
        k2_target = safe_divide((self.target_bone[5][1] - self.target_bone[6][1]) , (self.target_bone[5][0] - self.target_bone[6][0]))
        right_upper_arm_angle = calculate_angle(k1_current,k2_target)
        self.update_pose(right_upper_arm_angle,control_bone=5)
        write_rotate_angle(self.camera_position,'5',right_upper_arm_angle)

        #左大腿
        k1_current = safe_divide((self.pose[8][1] - self.pose[9][1]) , (self.pose[8][0] - self.pose[9][0]))
        k2_target = safe_divide((self.target_bone[8][1] - self.target_bone[9][1]) , (self.target_bone[8][0] - self.target_bone[9][0]))
        left_thigh = calculate_angle(k1_current,k2_target)
        self.update_pose(left_thigh,control_bone=8)
        write_rotate_angle(self.camera_position,'8',left_thigh)

        #右大腿
        k1_current = safe_divide((self.pose[11][1] - self.pose[12][1]) , (self.pose[11][0] - self.pose[12][0]))
        k2_target = safe_divide((self.target_bone[11][1] - self.target_bone[12][1]) , (self.target_bone[11][0] - self.target_bone[12][0]))
        right_thigh = calculate_angle(k1_current,k2_target)
        self.update_pose(right_thigh,control_bone=11)
        write_rotate_angle(self.camera_position,'11',right_thigh)
        
        self.ax_step2.draw_scatter(self.pose,color = 'g')
        '''
        旋转小臂,小腿
        '''
        #左小臂
        k1_current = safe_divide((self.pose[3][1] - self.pose[4][1]) , (self.pose[3][0] - self.pose[4][0]))
        k2_target = safe_divide((self.target_bone[3][1] - self.target_bone[4][1]) , (self.target_bone[3][0] - self.target_bone[4][0]))
        left_lower_arm_angle = calculate_angle(k1_current,k2_target)
        self.update_pose(left_lower_arm_angle,control_bone=3)
        write_rotate_angle(self.camera_position,'3',left_lower_arm_angle)

        #右小臂
        k1_current = safe_divide((self.pose[6][1] - self.pose[7][1]) , (self.pose[6][0] - self.pose[7][0]))
        k2_target = safe_divide((self.target_bone[6][1] - self.target_bone[7][1]) , (self.target_bone[6][0] - self.target_bone[7][0]))
        right_lower_arm_angle = calculate_angle(k1_current,k2_target)
        self.update_pose(right_lower_arm_angle,control_bone=6)
        write_rotate_angle(self.camera_position,'6',right_lower_arm_angle)

        #左小腿
        k1_current = safe_divide((self.pose[9][1] - self.pose[10][1]) , (self.pose[9][0] - self.pose[10][0]))
        k2_target = safe_divide((self.target_bone[9][1] - self.target_bone[10][1]) , (self.target_bone[9][0] - self.target_bone[10][0]))
        left_leg_angle = calculate_angle(k1_current,k2_target)
        self.update_pose(left_leg_angle,control_bone=9)
        write_rotate_angle(self.camera_position,'9',left_leg_angle)

        #右小腿
        k1_current = safe_divide((self.pose[12][1] - self.pose[13][1]) , (self.pose[12][0] - self.pose[13][0]))
        k2_target = safe_divide((self.target_bone[12][1] - self.target_bone[13][1]) , (self.target_bone[12][0] - self.target_bone[13][0]))
        right_leg_angle = calculate_angle(k1_current,k2_target)
        self.update_pose(right_leg_angle,control_bone=12)
        write_rotate_angle(self.camera_position,'12',right_leg_angle)

        self.ax_step3.draw_scatter(self.pose,color = 'g')
        self.ax_step3.draw_scatter(self.target_bone,color = 'b')
    #用于平移移动骨骼
    def move(self):
        x_move = next_pose_info[1][0] - self.pose[1][0]
        y_move = next_pose_info[1][1] - self.pose[1][1]
        #整体平移
        for i in range(len(self.pose)):
            self.pose[i][0] = self.pose[i][0] + x_move
            self.pose[i][1] = self.pose[i][1] + y_move




if __name__ == "__main__":
    
    initialize_pose_data = read_pose_data('AMC/output/pose_data.json')
    initialize_front_plot_list = initialize_pose_data[0]['front'][0]['initialize_pose_info']
    initialize_left_plot_list = initialize_pose_data[1]['left'][0]['initialize_pose_info']
    initialize_right_plot_list = initialize_pose_data[2]['right'][0]['initialize_pose_info']

    initialize_ax_front.draw_scatter(initialize_front_plot_list,color='r')

    
    for plot_index,plot in enumerate(initialize_front_plot_list):
        if plot_index in [2,3,4,8,9,10]:
            initialize_front_plot_list[plot_index].append(initialize_right_plot_list[plot_index][0])
            
        elif plot_index in [5,6,7,11,12,13]:
            initialize_front_plot_list[plot_index].append(initialize_left_plot_list[plot_index][0])
            
        else:
            initialize_front_plot_list[plot_index].append(initialize_right_plot_list[plot_index][0])
            
    initialize_ax_right.draw_scatter(initialize_right_plot_list,color='r')
    initialize_ax_left.draw_scatter(initialize_left_plot_list,color='r')

    for plot_index,plot in enumerate(initialize_front_plot_list):
        if plot_index in [2,3,4,8,9,10]:
            initialize_ax_3d.draw_scatter([plot],color='b')
        elif plot_index in [5,6,7,11,12,13]:
            initialize_ax_3d.draw_scatter([plot],color='r')
        else:
            initialize_ax_3d.draw_scatter([plot],color='g')

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

    initialize_plot_list = [initialize_front_plot_list,
                            initialize_left_plot_list,initialize_right_plot_list]
    ax = [front_ax_initialize,left_ax_initialize,right_ax_initialize]

    for initialize_index,initialize_list in enumerate(initialize_plot_list):
        #绘画初始化骨骼和目标骨骼图像
        ax[initialize_index].draw_scatter(initialize_list,color = 'r')
        ax[initialize_index].draw_scatter(next_pose_info_list[initialize_index],color = 'b')
        #移除第三项
        initialize_list = [[x[0], x[1]] for x in initialize_list]
        #print('depart:',initialize_list)
        #表示新的骨骼点18,19
        initialize_list.append([initialize_list[1][0]-1,initialize_list[1][1]])#18
        initialize_list.append([initialize_list[1][0]+1,initialize_list[1][1]])#19
        next_pose_info_list[initialize_index].append([next_pose_info_list[initialize_index][1][0]-1,
                                                      next_pose_info_list[initialize_index][1][1]])#18(target)
        next_pose_info_list[initialize_index].append([next_pose_info_list[initialize_index][1][0]+1,
                                                      next_pose_info_list[initialize_index][1][1]])#19(target)
        
        initialize_plot_list[initialize_index] = initialize_list


    front_armature = Armature(initialize_plot_list[0],front_ax_initialize,
                              front_ax_step1,front_ax_step2,front_ax_step3,'front')
    front_armature.move()
    front_armature.rotate(next_pose_info_list[0])

    left_armature = Armature(initialize_plot_list[1],left_ax_initialize,
                              left_ax_step1,left_ax_step2,left_ax_step3,'left')
    left_armature.move()
    left_armature.rotate(next_pose_info_list[1])

    right_armature = Armature(initialize_plot_list[2],right_ax_initialize,
                              right_ax_step1,right_ax_step2,right_ax_step3,'right')
    right_armature.move()
    right_armature.rotate(next_pose_info_list[2])

    plt.show(block=True)