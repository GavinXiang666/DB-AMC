from camera import *
import time

record_thread = []
camera_index_list = [0,1,2]
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