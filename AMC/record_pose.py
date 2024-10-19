from camera_setting import *
import os,csv

if __name__ == "__main__":
    camera_threads = []
    with open('AMC/camera_config.csv', 'r', newline='') as config_file:
        csvreader = csv.reader(config_file)
        for line in csvreader:
            camera_index_list = [line[0], line[1], line[2]] 
            camera_index_list = list(map(int, camera_index_list))
            
    for camera in camera_index_list:
        camera = Camera(camera,f'AMC/output/video/Camera {camera}.avi')
        #创建并启动摄像头线程
        camera_threads.append(CameraThread(camera))

    for thread in camera_threads:
        thread.start()

    # 等待 5 秒后发出录像信号
    print("等待 3 秒后开始录像...")
    time.sleep(3)

    # 触发开始录像事件
    start_event.set()

    # 等待所有线程结束
    for thread in camera_threads:
        thread.join()

    print("录像结束")