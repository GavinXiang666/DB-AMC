import cv2
import torch
import numpy as np
import traceback
import sys  
sys.path.insert(0, 'build_model')  
from src.body import Body  

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




