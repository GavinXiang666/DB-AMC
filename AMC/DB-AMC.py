import cv2,os
import tkinter as tk
from YoloPose import Openpose_Detector
from camera_setting import Camera
from tkinter import messagebox,filedialog
import tkinter.simpledialog as simpledialog
import subprocess
from PIL import Image, ImageTk
import pyfiglet,csv,sys
import ffmpeg

current_process = None

with open('AMC/camera_config.csv', 'w', newline='') as config_file:
    pass

def camera_setting():
    global current_process  
    if current_process:  
        current_process.kill()  # 停止当前进程
    print(pyfiglet.figlet_format("Camera Setting",font="slant"))
    camera_index_input = simpledialog.askstring("Input camera index", "Please enter the camera index, separated by a space:")
    camera_index_input = camera_index_input.split()

    with open('AMC/camera_config.csv', 'w', newline='') as config_file:
        config_writer = csv.writer(config_file)
        config_writer.writerow(camera_index_input)

    if camera_index_input:
        if len(camera_index_input) == 3:
            camera_index_list = [camera_index_input[0],camera_index_input[1],
                                 camera_index_input[2]]
            current_process = subprocess.Popen(["python", "AMC/camera_setting.py"])
        else:
            messagebox.showinfo("Warning", "Please check that the input is correct")
def openpose_test():
    global current_process  
    if current_process:  
        current_process.kill()  # 停止当前进程
    
    print(pyfiglet.figlet_format("Openpose Test",font="slant"))
    print("Loading the model......")
    openpose_detector = Openpose_Detector()
    print("Openpose loading successfully")

    camera_index_list = []
    with open('AMC/camera_config.csv', 'r', newline='') as config_file:
        csvreader = csv.reader(config_file)
        for line in csvreader:
            camera_index_list = [line[0], line[1], line[2]]
            camera_index_list = list(map(int, camera_index_list))
    
    test_image_list = []
    for index in camera_index_list:    
        camera = Camera(camera_index=index)
        image = camera.get_frame()
        test_image_list.append(image)
    for test_image in test_image_list:
        result_image,info = openpose_detector.detect_pose(test_image)
        cv2.imshow("openpose_test",result_image)
        cv2.waitKey(0)
def record_pose():
    print(pyfiglet.figlet_format("Record Pose",font="slant"))
    global current_process  
    if current_process:  
        current_process.kill()  # 停止当前进程
    current_process = subprocess.Popen(["python", "AMC/record_pose.py"])

def auto_clip_video():
    print(pyfiglet.figlet_format("Auto Clip Video",font="slant"))
    global current_process  
    if current_process:  
        current_process.kill()  # 停止当前进程

    selected_folder = filedialog.askdirectory()
    print("Folder Path:",selected_folder)
    time_input = simpledialog.askstring("Input Clip Time", "Please enter the retention time:")
    retention_time_input = float(time_input.split(" ")[1])
    start_time_input = float(time_input.split(" ")[0])

    avi_files = []
    for video_file in os.listdir(selected_folder):
        if video_file.endswith(".avi"):
            video_path = selected_folder+"/"+video_file
            avi_files.append(video_path)
    print(avi_files)
    #自动切片
    for input_video in avi_files:
        output_video = "/".join(input_video.rsplit('/', 1)[:-1] + [f"New {input_video.rsplit('/', 1)[-1]}"])
        
        ffmpeg.input(input_video, ss=start_time_input, t=retention_time_input).output(output_video).run()
        os.remove(path=input_video)
    
def stop_current_process():  
    print("")
    print("stop current process")
    global current_process  
    if current_process:  
        current_process.kill()  # 停止当前进程


if __name__ == "__main__":
    #注 入 灵 魂
    print(pyfiglet.figlet_format("Dream Busters!",font="slant"))

    English_Button_Text = ['Camera Setting','Openpose Test','Record Pose','Auto Clip Video','Bone Analysis','Stop Current Process']
    Chinese_Button_Text = ['摄像头设置','Openpose测试','录入姿态','视频自动切片','骨骼分析','停止当前进程']
    
    mode = "Chinese"
    if mode == "Chinese":
        language_text = Chinese_Button_Text
    else:
        language_text = English_Button_Text

    # 创建主窗口  
    tk_window = tk.Tk()  
    tk_window.title("Dream Busters! Automatic Motion Capture")  
    tk_window.geometry('450x600')
    
    original_image = Image.open("logo/logo_white.png")
    width, height = original_image.size  
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    resized_image = original_image.resize((new_width, new_height))

    logo = ImageTk.PhotoImage(resized_image)
    label = tk.Label(tk_window, image=logo)  
    label.pack()  
    # 创建按钮  
    button1 = tk.Button(tk_window, text=language_text[0], command=camera_setting)  
    button1.pack()  
    button2 = tk.Button(tk_window, text=language_text[1], command=openpose_test)  
    button2.pack()
    button3 = tk.Button(tk_window, text=language_text[2], command=record_pose)  
    button3.pack()

    button4 = tk.Button(tk_window, text=language_text[3], command=auto_clip_video)
    button4.pack()

    button5 = tk.Button(tk_window, text=language_text[4], command=openpose_test)  
    button5.pack()

    
    stop_button = tk.Button(tk_window, text=language_text[5], command=stop_current_process)  
    stop_button.pack()  

    # 运行主循环  
    tk_window.mainloop()  