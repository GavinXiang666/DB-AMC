import sqlite3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from functools import partial
import cv2

frames = 0
def update_vector(pitch,yaw,psi,vector):
    pitch,yaw = np.radians(pitch),np.radians(yaw)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R = Rx @ Ry @ Rz

    v_rotated = R @ vector
    return v_rotated
#坐标轴AX类型
class Ax:
    def __init__(self, layout, fig, projection=None):
        '''
        layout--->布局,figure的排列顺序
        fig--->figure
        projection--->'3d'
        '''
        self.projection = projection
        self.layout = layout
        self.fig = fig
        self.ax = self.fig.add_subplot(self.layout, projection=self.projection)
        self.ax.set_xlim(-600, 600)  
        self.ax.set_ylim(-600, 600)  
        if self.projection == '3d':
            self.ax.set_zlim(-600, 600)
    
    def set_title(self, title_name):
        self.ax.set_title(title_name)
    
    def draw_scatter(self, scatter, color):
        x, y, z = scatter
        self.ax.scatter(x, y, z, s=2, color=color)

    def draw_extra(self,pose_list):
        def draw_plot(start_scatter, end_scatter, color):
            x = [start_scatter[0], end_scatter[0]]
            y = [start_scatter[1], end_scatter[1]]
            z = [start_scatter[2], end_scatter[2]]
            
            self.ax.plot(x, y, z, color=color, linewidth=2)
        draw_plot(pose_list[0],pose_list[1],color='green')
        draw_plot(pose_list[0],pose_list[2],color='green')
        draw_plot(pose_list[1],pose_list[3],color='green')
        draw_plot(pose_list[2],pose_list[4],color='green')
        #draw_plot(pose_list[3],pose_list[5],color='green')
        #draw_plot(pose_list[4],pose_list[6],color='green')

        draw_plot(pose_list[5],pose_list[6],color='orange')
        draw_plot(pose_list[5],pose_list[7],color='orange')
        draw_plot(pose_list[6],pose_list[8],color='orange')
        draw_plot(pose_list[7],pose_list[9],color='orange')
        draw_plot(pose_list[8],pose_list[10],color='orange')

        draw_plot(pose_list[5],pose_list[11],color='purple')
        draw_plot(pose_list[6],pose_list[12],color='purple')
        draw_plot(pose_list[11],pose_list[12],color='purple')

        draw_plot(pose_list[11],pose_list[13],color='blue')
        draw_plot(pose_list[12],pose_list[14],color='blue')
        draw_plot(pose_list[13],pose_list[15],color='blue')
        draw_plot(pose_list[14],pose_list[16],color='blue')

        
        draw_plot((np.array(pose_list[5])+np.array(pose_list[6]))/2,
                  (np.array(pose_list[11])+np.array(pose_list[12]))/2,color="red")
        draw_plot((np.array(pose_list[3])+np.array(pose_list[4]))/2,
                  (np.array(pose_list[5])+np.array(pose_list[6]))/2,color="red")

    def text(self,x,y,z,text):
        self.ax.text(x, y, z,text, color='black', 
                     fontsize=4, fontweight='bold')
    
    def draw_xyz(self,axis_length):
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', linewidth=2, label='X', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', linewidth=2, label='Y', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', linewidth=2, label='Z', arrow_length_ratio=0.1)

    def clear(self):
        self.ax.clear()

#绘制pose
def draw_pose(ax, pose_list,id_list,mode="Normal"):
    '''
    ax: Ax对象
    pose_list: 当前帧的姿态点列表
    mode: 绘制模式
    '''
    if mode == "Normal":
        for point_index,point in enumerate(pose_list):
            ax.draw_scatter(point, color='blue')
            ax.text(point[0], point[1], point[2],id_list[point_index])
        ax.draw_extra(pose_list)
if __name__ == "__main__":
    '''
    连接sqlite数据库
    '''
    conn = sqlite3.connect("sql/pose_database.db")
    cursor = conn.cursor()
    conn2 = sqlite3.connect("sql/pose_detect_result_database.db")
    cursor2 = conn2.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables2 = cursor2.fetchall()


    all_frames = []
    all_id = []
    for table in tables:
        cursor.execute(f"SELECT * FROM {table[0]};")
        rows = cursor.fetchall()
        
        #坐标
        frame_data = []
        id_data = []
        for i,row in enumerate(rows):
            # v = np.array([float(row[1]), float(row[2]), float(row[3])])
            # x,y,z = update_vector(-90,-90,0,v)
            frame_data.append([float(row[1]), float(row[2]), float(row[3])])
            id_data.append(str(row[0]))
        
        
        all_frames.append(frame_data)
        all_id.append(id_data)

    images = []
    
    for table2 in tables2:
        cursor2.execute(f"SELECT * FROM {table2[0]};")
        rows2 = cursor2.fetchall()
        #图片
        resize_image = []
        for row2 in rows2:
            # 将二进制数据转换为 numpy 数组
            nparr = np.frombuffer(row2[1], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为 OpenCV 图像
            img_resized = cv2.resize(img, (480,480))
            resize_image.append(img_resized)
    
        concatenated_img = np.hstack(resize_image)
        images.append(concatenated_img)
    
    #创建图形和坐标轴
    pose_fig = plt.figure(num="pose figure", figsize=(6, 6))
    pose_ax = Ax(111, pose_fig, '3d')
    pose_ax.set_title('Pose Animation 3D')
    

    #动画更新
    def update(frame,id):
        global frames
        if frames <= len(images):

            pose_ax.clear()
            pose_ax.draw_xyz(axis_length=550)
            pose_ax.set_title(f'Pose Frame {frame+1}/{len(all_frames)}')
            draw_pose(pose_ax, all_frames[frame],all_id[frame], mode="Normal")

            img = images[frames-1]
            cv2.imshow("detect_result", img)
            #cv2.waitKey(700)
            frames += 1
            return [pose_ax.ax]
            
        else:
            frames = 0
    
    #创建动画
    ani = FuncAnimation(
        pose_fig, 
        partial(update,id=id_data), 
        frames=len(all_frames),

        interval=200,#帧间隔
        blit=False
    )
    
    plt.show()
    conn.close()