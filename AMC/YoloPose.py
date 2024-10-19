import cv2
import torch
import numpy as np
import traceback
import sys  
sys.path.insert(0, 'build_model')  
from src.body import Body  

class Yolo_Detector:
    def __init__(self,device):
        self.device = device
        self.model = torch.hub.load('yolov5', 'custom',
                                'yolov5/yolov5s.pt',source='local', force_reload=True)
    def detect_person(self,frame):
        frame_copy = frame.copy()
        #指定模型在cuda上运行
        self.model = self.model.to(self.device)
        results = self.model(frame)
        results.print()

        count = 0
        person_list = []
        try:
            xmins = results.pandas().xyxy[0]['xmin']#获取所有的xmin坐标
            ymins = results.pandas().xyxy[0]['ymin']#获取所有的ymin坐标
            xmaxs = results.pandas().xyxy[0]['xmax']#获取所有的xmax坐标
            ymaxs = results.pandas().xyxy[0]['ymax']#获取所有的ymax坐标
            class_list = results.pandas().xyxy[0]['class']#获取类别信息(人的类别信息为0)
            confidences = results.pandas().xyxy[0]['confidence']#获取信任度
            for xmin, ymin, xmax, ymax,class_l,conf in zip(xmins, ymins, xmaxs, ymaxs,class_list,confidences):#for循环遍历
                #print("detect_armor_class:", detect_armor_class)
                if class_l == 0 and conf>=0.3:#如果识别类别为人且置信度大于0.3
                    single_person_information = {
                        "id":count,
                        "position":[int(xmin), int(ymin), int(xmax), int(ymax)],
                        "area":(int(xmax)-int(xmin)) * (int(ymax)-int(ymin))
                    }
                    person_list.append(single_person_information)
                    count = count + 1

            #找到距离摄像机最近的人
            target_person = max(person_list, key=lambda x: x["area"])
            position = target_person["position"]
            x, y, w, h = position[0], position[1], position[2], position[3]
            #提取人物ROI区域
            target_person_roi = frame[y:h+10,x-10:w+10]

            frame_copy = cv2.rectangle(frame_copy, (x, y), (w, h), (0, 255, 0), 3)
            frame_copy = cv2.putText(frame_copy, "target_person", (x, y-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0), 1, cv2.LINE_AA)
            
            #cv2.imshow("frame",frame)
            #cv2.imshow("yolo_detect",frame_copy)
            cv2.destroyAllWindows()
            return target_person_roi
        except:
            traceback.print_exc()
class Openpose_Detector:
    def __init__(self):
        self.model = Body('build_model/model/body_pose_model.pth')

    def detect_pose(self,frame):
        candidate, subset = self.model(frame)
        body_pose_information = []
        for person in subset.astype(int):
            #print(person)
            for i in range(0,18):
                node_index = person[i]
                #print(node_index)
                x1,y1 = candidate[node_index][:2]
                #print(x1,y1)
            #print("candidate"+str(candidate))
            #print(subset)
                body_pose_information.append((x1,y1))
            #print(all_pose_peaks)
        for pose_index, (x, y) in enumerate(body_pose_information):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(pose_index), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)  # 在点的上方显示索引
            
        
        #cv2.imshow("openpose_detect",frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return [frame,body_pose_information]
if __name__ == "__main__":
    yolo_detector = Yolo_Detector("cuda")
    openpose_detector = Openpose_Detector()

    image = cv2.imread("build_model/images/man1.jpeg")

    yolo_image = yolo_detector.detect_person(image)
    initialize_info = openpose_detector.detect_pose(yolo_image)
    initialize_pose_image = initialize_info[0]
    initialize_pose_info = initialize_info[1]
    print(f'initialize_pose_info: {initialize_pose_info}')

    cv2.imshow("openpose_detect",initialize_pose_image)
    cv2.waitKey(0)

    cv2.destroyWindow("frame_copy")  
    cv2.destroyWindow("openpose_detect") 
    cv2.destroyAllWindows()  



