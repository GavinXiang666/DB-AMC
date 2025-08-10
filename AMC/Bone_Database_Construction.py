import os,time
from datetime import datetime
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from loguru import logger
from yolo_keypoints import *
from tqdm import tqdm

logger.add("AMC/log/Bone_Database_Construction.log",rotation="10 MB")
logger.info(f"Start Time: {datetime.now()}")
'''
创建数据库
'''
try:
    os.remove('sql/pose_database_2d.db')
    os.remove('sql/pose_database.db')
    os.remove('sql/pose_detect_result_database.db')
except:
    logger.info("Create a new database......")

'''
连接sqlite数据库
'''
conn = sqlite3.connect("sql/pose_database_2d.db")
cursor = conn.cursor()

image_conn = sqlite3.connect("sql/pose_image_database.db")
image_cursor = image_conn.cursor()

detect_result_conn = sqlite3.connect("sql/pose_detect_result_database.db")
detect_result_cursor = detect_result_conn.cursor()

"""加载模型"""
DB_detector = Keypoints_Detector()
logger.info("Yolov7-Pose Model is successfully loaded")

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

class GPUMonitor:
    def __init__(self):
        self.peak_allocated = 0
    
    def __call__(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            self.peak_allocated = max(self.peak_allocated, allocated)
            total = torch.cuda.get_device_properties(0).total_memory
            
            print(
                f"GPU: {allocated/1024**3:.1f}GB/{total/1024**3:.0f}GB "
                f"({allocated/total*100:.0f}%) | "
                f"Peak: {self.peak_allocated/1024**3:.1f}GB",
                end='\r'
            )
monitor = GPUMonitor()

'''
视频处理,骨骼分析类
'''
class Dream_Eyes:
    def __init__(self,detect_mode=None,front_video=None,left_video=None,right_video=None):
        self.eye = DB_detector
        self.detect_mode = detect_mode
        if self.detect_mode == "Video":
            logger.info("Detect Mode: Video")
            self.front_video = front_video
            self.left_video = left_video
            self.right_video = right_video
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

        elif self.detect_mode == "Images":
            logger.info("Detect Mode: Images")
            
    def view(self,frame_index=None,table_name=None):
        view_result = []
        if self.detect_mode == "Video":
            detector_sets=[
                self.front_detector,
                self.left_detector,
                self.right_detector
            ]
            for cap_index,cap in enumerate(detector_sets):
                #设置视频到指定帧数
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, particular_frame = cap.read()
                if ret == True:
                    view_result.append(particular_frame)#添加当前帧
                else:
                    logger.error(f"View Failed.--->frame_index: {frame_index} cap_index: {cap_index}")
            
            return view_result
        elif self.detect_mode == "Images":

            image_cursor.execute(f"SELECT * FROM {table_name};")
            rows = image_cursor.fetchall()
            
            for image_index,row in enumerate(rows):
                #图片二进制数据
                image_data = row[1]
                #二进制数据转numpy数组
                nparr = np.frombuffer(image_data, np.uint8)
                #解码
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                view_result.append(image)#Front,Left,Right
            
            return view_result
    '''
    骨骼识别
    '''
    def detect_pose(self,view):
        #禁用梯度计算防止爆显存
        with torch.no_grad():
            pose_data = []
            image_data = []
            #顺序为front-left-right
            for view_index,view_image in enumerate(view):
                #[0]为骨骼识别信息,[1]为图像
                pose_data_info,result_image = self.eye.detect_pose(view_image)
                pose_data.append(pose_data_info)
                image_data.append(result_image)
        return [pose_data,image_data]

def find(table_name,keypoint_name):
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    x_value = None;y_value = None
    for row in rows:
        if row[0] == keypoint_name:
            x_value = row[1]
            y_value = row[2]
            break
    return [x_value,y_value]

'''
Video视频模式
'''

def Video(folder_path):
    #关闭用不上的数据库
    image_cursor.close()
    image_conn.close()     
    #抽帧常数
    frame_extraction = 1
    logger.info(f"Frame Extraction: {frame_extraction}")
    #DreamBusters Eyes
    Dream_Eye = Dream_Eyes(detect_mode="Video",front_video=r"AMC\output\video\New New New Camera 2.avi",
                            left_video=r"AMC\output\video\New New New Camera 3.avi",right_video=r"AMC\output\video\New New New Camera 0.avi")
    now_time = time.time()

    #根据抽帧常数构建原始2d数据库
    for frame_index in tqdm(range(0,int(Dream_Eye.total_frame_count // frame_extraction)),desc='Detect Pose'):
    #for frame_index in tqdm(range(0,2),desc='Detect Pose'):
        #获取图像
        view_result = Dream_Eye.view(frame_index=frame_index)
        pose_result = Dream_Eye.detect_pose(view_result)
        monitor()

        #创建Table(存识别结果图片)
        detect_result_cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {f'frame_{frame_index}'} (
                name TEXT NOT NULL,
                image_data BLOB NOT NULL
            )
        ''')
        detect_result_conn.commit()
        
        for result_index,detect_result_image in enumerate(pose_result[1]):
            _, img_encoded = cv2.imencode('.png', detect_result_image)
            img_bytes = img_encoded.tobytes()
            
            name = ['front','left','right'][result_index]
            table_name = f'frame_{frame_index}'

            detect_result_cursor.execute(f'INSERT INTO {table_name} (name, image_data) VALUES (?, ?)', (name, img_bytes))
            detect_result_conn.commit()

        #每一帧构建三张表
        for side_index,side in enumerate(['front','left','right']):
            try:
                #创建Table
                command_string = '''CREATE TABLE {}_frame_{}(
                ID VARCHAR PRIMARY KEY NOT NULL,
                X  INT NOT NULL,
                Y  INT NOT NULL,
                Conf  DOUBLE NOT NULL
                );'''.format(side,frame_index)
                cursor.execute(command_string)
            #仅限sqlite数据库操作错误
            except sqlite3.OperationalError:
                logger.error(f"Build Table Failed(OperationalError)--->TABLE {side}_frame_{frame_index}")

            for keypoint in range(0,17):
                id_value = str(pose_result[0][side_index][keypoint]['name'])
                x_value = pose_result[0][side_index][keypoint]['data'][0]
                y_value = pose_result[0][side_index][keypoint]['data'][1]
                conf_value = pose_result[0][side_index][keypoint]['data'][2]

                #往Table里插入数据
                insert_command = f"""
                INSERT INTO {side}_frame_{frame_index} 
                (ID, X, Y, Conf) 
                VALUES (?, ?, ?, ?);
                """
                cursor.execute(insert_command, (id_value, x_value, y_value, conf_value))
                conn.commit()

    logger.info("Build database completely")
    logger.info(f"Total time: {time.time()-now_time}s")

    '''
    根据2d数据库重建新的数据库
    '''
    logger.warning("<--- Database building, do not quit --->")

    new_conn = sqlite3.connect("sql/pose_database.db")
    new_cursor = new_conn.cursor()
    #获取原2d数据库的所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_2d = cursor.fetchall()

    for k in range(0,len(tables_2d)//3):
        command_string = '''CREATE TABLE frame_{}(
                ID VARCHAR PRIMARY KEY NOT NULL,
                X  INT NOT NULL,
                Y  INT NOT NULL,
                Z INT NOT NULL
                );'''.format(k)
        new_cursor.execute(command_string)

    for n in range(0,len(tables_2d)//3):
        for keypoint_name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                            'left_shoulder', 'right_shoulder', 'left_elbow', 
                            'right_elbow', 'left_wrist', 'right_wrist', 
                            'left_hip', 'right_hip', 'left_knee', 
                            'right_knee', 'left_ankle', 'right_ankle']:
            x_value,y_value = find(f'front_frame_{n}',keypoint_name)
            if keypoint_name == "nose":
                z_value = find(f'right_frame_{n}',keypoint_name)[0]
            elif keypoint_name.split("_")[0] == "left":
                z_value = find(f'left_frame_{n}',keypoint_name)[0]
            elif keypoint_name.split("_")[0] == "right":
                #转一下坐标系
                z_value = 960 - find(f'right_frame_{n}',keypoint_name)[0]
            
            #应用旋转对应Blender
            v = np.array([x_value,y_value,z_value])
            x_value,y_value,z_value = update_vector(-90,-90,0,v)

            insert_command = f"INSERT INTO frame_{n} (ID, X, Y, Z) VALUES ('{keypoint_name}', {x_value}, {y_value},{z_value});"
            
            new_cursor.execute(insert_command)
            new_conn.commit()
    logger.info("Build new 3d databse successfully")

    """
    关闭数据库连接
    """
    cursor.close()
    conn.close()
    new_cursor.close()
    new_conn.close()
    detect_result_cursor.close()
    detect_result_conn.close()
    logger.info("The database has been closed")

'''
Images图片集模式
'''

def Images(fodler_path):
    Dream_Eye = Dream_Eyes(detect_mode="Images")
    now_time = time.time()

    image_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    frame_tables = image_cursor.fetchall()
    
    
    for frame_index in tqdm(range(0,len(frame_tables)),desc='Detect Pose'):
        view_result = Dream_Eye.view(table_name=frame_tables[frame_index][0])
        pose_result = Dream_Eye.detect_pose(view_result)
        monitor()

        #创建Table(存识别结果图片)
        detect_result_cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {f'frame_{frame_index}'} (
                name TEXT NOT NULL,
                image_data BLOB NOT NULL
            )
        ''')
        detect_result_conn.commit()

        for result_index,detect_result_image in enumerate(pose_result[1]):
            _, img_encoded = cv2.imencode('.png', detect_result_image)
            img_bytes = img_encoded.tobytes()
            
            name = ['front','left','right'][result_index]
            table_name = f'frame_{frame_index}'

            detect_result_cursor.execute(f'INSERT INTO {table_name} (name, image_data) VALUES (?, ?)', (name, img_bytes))
            detect_result_conn.commit()
        
        #每一帧构建三张表
        for side_index,side in enumerate(['front','left','right']):
            try:
                #创建Table
                command_string = '''CREATE TABLE {}_frame_{}(
                ID VARCHAR PRIMARY KEY NOT NULL,
                X  INT NOT NULL,
                Y  INT NOT NULL,
                Conf  DOUBLE NOT NULL
                );'''.format(side,frame_index)
                cursor.execute(command_string)

            #仅限sqlite数据库操作错误
            except sqlite3.OperationalError:
                logger.error(f"Build Table Failed(OperationalError)--->TABLE {side}_frame_{frame_index}")

            for keypoint in range(0,17):
                id_value = str(pose_result[0][side_index][keypoint]['name'])
                x_value = pose_result[0][side_index][keypoint]['data'][0]
                y_value = pose_result[0][side_index][keypoint]['data'][1]
                conf_value = pose_result[0][side_index][keypoint]['data'][2]

                #往Table里插入数据
                insert_command = f"""
                INSERT INTO {side}_frame_{frame_index} 
                (ID, X, Y, Conf) 
                VALUES (?, ?, ?, ?);
                """
                cursor.execute(insert_command, (id_value, x_value, y_value, conf_value))
                conn.commit()

    logger.info("Build database completely")
    logger.info(f"Total time: {time.time()-now_time}s")

    '''
    根据2d数据库重建新的数据库
    '''
    logger.warning("<--- Database building, do not quit --->")

    new_conn = sqlite3.connect("sql/pose_database.db")
    new_cursor = new_conn.cursor()
    #获取原2d数据库的所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_2d = cursor.fetchall()

    for k in range(0,len(tables_2d)//3):
        command_string = '''CREATE TABLE frame_{}(
                ID VARCHAR PRIMARY KEY NOT NULL,
                X  INT NOT NULL,
                Y  INT NOT NULL,
                Z INT NOT NULL
                );'''.format(k)
        new_cursor.execute(command_string)

    for n in range(0,len(tables_2d)//3):
        for keypoint_name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                            'left_shoulder', 'right_shoulder', 'left_elbow', 
                            'right_elbow', 'left_wrist', 'right_wrist', 
                            'left_hip', 'right_hip', 'left_knee', 
                            'right_knee', 'left_ankle', 'right_ankle']:
            x_value,y_value = find(f'front_frame_{n}',keypoint_name)
            if keypoint_name == "nose":
                z_value = find(f'right_frame_{n}',keypoint_name)[0]
            elif keypoint_name.split("_")[0] == "left":
                z_value = find(f'left_frame_{n}',keypoint_name)[0]
            elif keypoint_name.split("_")[0] == "right":
                #转一下坐标系
                z_value = 960 - find(f'right_frame_{n}',keypoint_name)[0]
                
            #应用旋转对应Blender
            v = np.array([x_value,y_value,z_value])
            x_value,y_value,z_value = update_vector(-90,-90,0,v)

            insert_command = f"INSERT INTO frame_{n} (ID, X, Y, Z) VALUES ('{keypoint_name}', {x_value}, {y_value},{z_value});"
            
            new_cursor.execute(insert_command)
            new_conn.commit()
        
    logger.info("Build new 3d databse successfully")

    """
    关闭数据库连接
    """
    new_cursor.close()
    new_conn.close()
    detect_result_cursor.close()
    detect_result_conn.close()
    logger.info("The database has been closed")
'''
Tkinter图形界面
'''
class ModeSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("模式选择")
        self.root.geometry("400x300")
        
        self.selected_mode = None
        self.selected_folder = None

        #创建UI
        self.create_widgets()
    
    def create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="请选择模式", font=('Arial', 12)).pack(pady=10)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Video", 
                  command=lambda: self.select_mode("Video")).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Images", 
                  command=lambda: self.select_mode("Images")).pack(side=tk.LEFT, padx=10)
        
        self.result_frame = ttk.LabelFrame(main_frame, text="选择结果", padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.mode_label = ttk.Label(self.result_frame, text="未选择模式")
        self.mode_label.pack(anchor="w")
        
        self.folder_label = ttk.Label(self.result_frame, text="未选择文件夹")
        self.folder_label.pack(anchor="w")
        
        ttk.Button(main_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, pady=10)
    
    def select_mode(self, mode):
        """选择模式并选择文件夹"""
        self.selected_mode = mode
        self.mode_label.config(text=f"已选择模式: {mode}")
        
        folder_path = filedialog.askdirectory(title=f"请选择{mode}文件夹")
        if folder_path:
            self.selected_folder = folder_path
            self.folder_label.config(text=f"文件夹位置: {folder_path}")
            
            print(f"用户选择了{mode}模式")
            print(f"文件夹位置: {folder_path}")
            
            if mode == "Video":
                Video(folder_path=folder_path)
            elif mode == "Images":
                Images(fodler_path=folder_path)
        else:
            self.selected_mode = None
            self.mode_label.config(text="未选择模式")
            messagebox.showwarning("警告", "您没有选择文件夹")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModeSelectionApp(root)
    root.mainloop()
        
    
    