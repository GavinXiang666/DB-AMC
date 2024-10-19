
from loguru import logger
import pyfiglet,json,os
from camera import *
from YoloPose import *


if __name__ == "__main__":
    #创建程序日志
    logger.add("AMC/log/main.log",rotation="10 MB")
    logger.info("日志创建成功,程序开始") 
    with open('AMC/pose_data.json', 'w') as f:  
        pass
    pose_file = open('AMC/pose_data.json', 'w')
    #注入灵魂
    print("#"*80)
    print(pyfiglet.figlet_format("Dream Busters!",font="slant"))
    print(pyfiglet.figlet_format("M a i n",font="slant"))
    print("#"*80)
    
    camera_index_list = [1,2,3]
    
    camera_front = Camera(camera_index=camera_index_list[0])
    camera_left = Camera(camera_index=camera_index_list[1])
    camera_right = Camera(camera_index=camera_index_list[2])
    camera_objects = [camera_front,camera_left,camera_right]
    logger.info("摄像头测试正常")

    logger.info("进行人工对位7秒钟时间")
    show_thread = []
    for camera in camera_objects:
        thread = threading.Thread(target=camera.Shoot_Video, args=(640, 480, False))
        show_thread.append(thread)
        thread.start()
        
    for thread in show_thread:
        thread.join()

    #拍摄定位图
    image_front = camera_front.Get_Frame()
    image_left = camera_left.Get_Frame()
    image_right = camera_right.Get_Frame()
    camera_frame = [image_front,image_left,image_right]
    cv2.imwrite("AMC/output/image_front.jpg",image_front)
    cv2.imwrite("AMC/output/image_left.jpg",image_left)
    cv2.imwrite("AMC/output/image_right.jpg",image_right)
    #加载yolov5s和openpose模型
    logger.info("正在载入模型")
    yolo_detector = Yolo_Detector("cuda")
    logger.info("yolov5s模型加载成功")
    openpose_detector = Openpose_Detector()
    logger.info("openpose模型加载成功")

    logger.info("开始进行初始化动作骨骼识别")
    pose_data_list = []
    for i,image in enumerate(camera_frame):
        yolo_image = yolo_detector.detect_person(image)
        initialize_info = openpose_detector.detect_pose(yolo_image)
        initialize_pose_info = initialize_info[1]
        print(f'initialize_pose_info: {initialize_pose_info}')

        position = "front" if i == 0 else ("left" if i == 1 else "right") 
        pose_data = {
            position:[
                {"initialize_pose_info":initialize_pose_info}
            ]
        }
        pose_data_list.append(pose_data)
    #写入初始pose数据
    json.dump(pose_data_list, pose_file,indent=4)

    pose_file.close()
    print("初始化窗口即将关闭")
    cv2.waitKey(1000)
    logger.info("初始化完成")
    
    logger.info("5秒后开始捕捉")
    time.sleep(5)

    image_front = camera_front.Get_Frame()
    image_left = camera_left.Get_Frame()
    image_right = camera_right.Get_Frame()
    cv2.imwrite("AMC/output/image/image_front.jpg",image_front)
    cv2.imwrite("AMC/output/image/image_left.jpg",image_left)
    cv2.imwrite("AMC/output/image/image_right.jpg",image_right)
    logger.info("捕捉完成")
        