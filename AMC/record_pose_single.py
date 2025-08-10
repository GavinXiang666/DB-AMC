import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import csv
import sqlite3
from loguru import logger
import numpy as np
from camera_setting import *

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DreamBusters!            Record_Pose_Single")
        self.root.geometry("1200x800")
        
        #初始化数据库
        self.init_database()
        
        #加载相机配置
        self.load_camera_config()
        
        #初始化相机
        self.cameras = [Camera(idx) for idx in self.camera_index_list]
        self.current_frame_index = 0
        self.captured_images = {i: None for i in range(3)}
        
        #创建UI
        self.create_widgets()
    
    def init_database(self):
        """初始化数据库"""
        try:
            os.remove('sql/pose_image_database.db')
        except:
            logger.info("Create a new database......")
        
        self.conn = sqlite3.connect("sql/pose_image_database.db")
        self.cursor = self.conn.cursor()
    
    def load_camera_config(self):
        """加载相机配置"""
        try:
            with open('AMC/camera_config.csv', 'r', newline='') as config_file:
                csvreader = csv.reader(config_file)
                for line in csvreader:
                    self.camera_index_list = [int(line[0]), int(line[1]), int(line[2])]
        except Exception as e:
            logger.error(f"Failed to load the camera configuration: {e}")
            self.camera_index_list = [0, 1, 2]#默认值
    
    def create_widgets(self):
        """创建界面组件"""

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        

        self.status_var = tk.StringVar(value=f"Current frame: 0")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        

        self.capture_btn = ttk.Button(btn_frame, text="Record", command=self.capture_images)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.confirm_btn = ttk.Button(btn_frame, text="Save", command=self.confirm_save, state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.LEFT, padx=5)
        
        self.discard_btn = ttk.Button(btn_frame, text="Cancel", command=self.discard_images, state=tk.DISABLED)
        self.discard_btn.pack(side=tk.LEFT, padx=5)
        
        self.exit_btn = ttk.Button(btn_frame, text="Exit", command=self.exit_app)
        self.exit_btn.pack(side=tk.RIGHT, padx=5)
        
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.camera_labels = []
        for i, side in enumerate(['Front_Camera', 'Left_Camera', 'Right_Camera']):
            frame = ttk.LabelFrame(self.image_frame, text=side)
            frame.grid(row=0, column=i, padx=10, pady=5, sticky="nsew")
            self.image_frame.columnconfigure(i, weight=1)
            
            label = ttk.Label(frame)
            label.pack(padx=10, pady=10)
            self.camera_labels.append(label)
    
    def capture_images(self):
        """拍摄照片并显示"""
        #从三个相机获取图像
        for i, camera in enumerate(self.cameras):
            frame = camera.get_frame()
            
            # if i != 0:
            #     frame = cv2.resize(frame,(900,340))
            self.captured_images[i] = frame
            #显示图像
            self.display_image(frame, i)
        
        self.confirm_btn.config(state=tk.NORMAL)
        self.discard_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
    
    def display_image(self, frame, camera_idx):
        """在UI中显示图像"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(img)
        
        self.camera_labels[camera_idx].photo = photo
        self.camera_labels[camera_idx].config(image=photo)
    
    def confirm_save(self):
        """确认保存当前拍摄的照片"""
        #创建数据库表
        table_name = f"frame_{self.current_frame_index}"
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                name TEXT NOT NULL,
                image_data BLOB NOT NULL
            )
        ''')
        self.conn.commit()
        
        #保存照片到数据库
        for i, side in enumerate(['front', 'left', 'right']):
            self.insert_image(table_name, side, self.captured_images[i])
        
        logger.success(f"Frame {self.current_frame_index} saved!")
        
        #准备下一帧
        self.current_frame_index += 1
        self.status_var.set(f"Current frame: {self.current_frame_index}")
        self.reset_ui_for_next_capture()
    
    def insert_image(self, table_name, name, image):
        """将图像保存到数据库"""
        _, img_encoded = cv2.imencode('.png', image)
        img_bytes = img_encoded.tobytes()
        self.cursor.execute(f'INSERT INTO {table_name} (name, image_data) VALUES (?, ?)', (name, img_bytes))
        self.conn.commit()
    
    def discard_images(self):
        """丢弃当前拍摄的照片"""
        self.captured_images = {i: None for i in range(3)}
        self.reset_ui_for_next_capture()
        logger.info("The operation has been cancelled")
    
    def reset_ui_for_next_capture(self):
        """重置UI进行下一次拍摄"""
        for label in self.camera_labels:
            label.config(image='')
            label.photo = None
        
        self.confirm_btn.config(state=tk.DISABLED)
        self.discard_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
    
    def exit_app(self):
        """退出应用程序"""
        if messagebox.askyesno("Exit", "Sure to quit?"):
            self.cursor.close()
            self.conn.close()
            
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()