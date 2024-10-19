import cv2
import threading
import time
import csv

# 全局同步事件
start_event = threading.Event()

class Camera:
    def __init__(self, camera_index, output_file=None):
        self.camera_index = camera_index
        self.output_file = output_file
        self.cap = cv2.VideoCapture(self.camera_index)
        self.is_recording = True
        self.is_displaying = True
        
        if self.cap.isOpened():
            print(f"Camera {self.camera_index} initialized successfully.")
        else:
            print(f"Failed to initialize camera {self.camera_index}.")
    
    # 开始录像的函数，等待全局同步事件
    def start_recording(self):
        # 等待同步的开始事件
        start_event.wait()

        # 设置录像格式和输出
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_file, fourcc, 20.0, (640, 480))
        
        # 开始录像
        while self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                out.write(frame)
                cv2.imshow(f'Camera {self.camera_index}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_recording = False
            else:
                break
        
        # 释放资源
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

    # 显示摄像头画面的函数，不保存录像
    def start_displaying(self):
        while self.is_displaying:
            ret, frame = self.cap.read()
            if ret:
                cv2.line(frame, (320, 0), (320, 480), (0, 0, 255), 2)
                cv2.imshow(f'Live Camera {self.camera_index}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_displaying = False
            else:
                break

        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
# 创建摄像头对象的线程
class CameraThread(threading.Thread):
    def __init__(self, camera, mode="record"):
        super().__init__()
        self.camera = camera
        self.mode = mode  # "record" for recording, "display" for just displaying

    def run(self):
        if self.mode == "record":
            self.camera.start_recording()
        elif self.mode == "display":
            self.camera.start_displaying()

if __name__ == "__main__":
    camera_threads = []
    
    # 读取摄像头配置
    with open('AMC/camera_config.csv', 'r', newline='') as config_file:
        csvreader = csv.reader(config_file)
        for line in csvreader:
            camera_index_list = [line[0], line[1], line[2]] 
            camera_index_list = list(map(int, camera_index_list))

    # 创建摄像头对象和线程
    for camera_index in camera_index_list:
        camera = Camera(camera_index)  # 暂时不传递 output_file，因为不录象
        # 创建并启动显示线程（不录像，只显示）
        camera_threads.append(CameraThread(camera, mode="display"))

    # 启动显示线程
    for thread in camera_threads:
        thread.start()

    # 等待所有显示线程结束
    for thread in camera_threads:
        thread.join()
