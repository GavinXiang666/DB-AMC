import cv2
import numpy as np
import time,os
import threading
import ffmpeg

class Camera:
    def __init__(self,camera_index):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.time = None
        if self.cap.isOpened():
            print(f"Camera {self.camera_index} initialized successfully.")
        else:
            print(f"Failed to initialize camera {self.camera_index}.")
    
    def Get_Frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            return frame
        else:
            print("Failed to take photo")

        return None
    def Shoot_Video(self,frame_width=640, frame_height=480,record_mode=True):
        if self.cap is not None:
            if record_mode == True:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output= cv2.VideoWriter(f'AMC/output/video/Camera {self.camera_index}.mp4',fourcc,30,(frame_width,frame_height))
                while self.cap.isOpened():
                    ret,frame = self.cap.read()

                    if not ret:
                        break

                    output.write(frame)
                    cv2.imshow(f"Camera {self.camera_index}",frame)

                    if cv2.waitKey(1) == ord('q'):
                        break
            elif record_mode == False:
                start_time = time.time()
                while self.cap.isOpened() and (time.time() - start_time) < 7:#7秒对位时间
                    ret,frame = self.cap.read()

                    if not ret:
                        break

                    cv2.imshow(f"Camera {self.camera_index}",frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

def trim_video(input_file, output_file, start_time, end_time):
    (
        ffmpeg
        .input(input_file, ss=start_time, t=end_time - start_time)
        .output(output_file, codec='copy')
        .run(overwrite_output=True)
    )

if __name__ == "__main__":
    record_thread = []
    camera_index_list = [1,2,0]
    for cap_index in camera_index_list:
        camera = Camera(camera_index=cap_index)
        thread = threading.Thread(target=camera.Shoot_Video)
        record_thread.append(thread)
        thread.start()
        time.sleep(0.05)
        if cap_index == camera_index_list[0]:
            first_camera_start_time = time.time()+0.07#记录第一个开启摄像头的时间
        elif cap_index == camera_index_list[1]:
            middle_camera_start_time = time.time()#记录第二个开启摄像头的时间
        else:
            latest_camera_start_time = time.time()#最后摄像头开启时间

    print(latest_camera_start_time)
    closed_time = None

    for i,thread in enumerate(record_thread):
        thread.join()
        if i == 0:
            closed_time = time.time()
    
    end_time = closed_time - latest_camera_start_time

    for video_index in range(len(camera_index_list)):
        input_video_path = f'AMC/output/video/Camera {video_index}.mp4'
        output_video_path = f'AMC/output/video/New_Camera {video_index}.mp4'
        print(video_index)
        if video_index == len(camera_index_list) - 1:
            trim_video(input_file=input_video_path, output_file=output_video_path,
                        start_time=0, end_time=end_time)
        elif video_index == len(camera_index_list) - 2:
            
            trim_video(input_file=input_video_path, output_file=output_video_path,
                        start_time=middle_camera_start_time - first_camera_start_time, end_time=end_time)
        else:
            trim_video(input_file=input_video_path, output_file=output_video_path,
                        start_time=latest_camera_start_time - first_camera_start_time,
                          end_time=end_time)
            
        os.remove(f'AMC/output/video/Camera {video_index}.mp4')#删除未处理的视频文件

    print("The End")
