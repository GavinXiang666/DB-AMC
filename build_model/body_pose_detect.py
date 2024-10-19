import cv2
import matplotlib.pyplot as plt
import copy,socket
import math,json
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand




def detect_pose(oriImg):
    body_estimation = Body('build_model/model/body_pose_model.pth')
    candidate, subset = body_estimation(oriImg)
    all_pose_peaks = []
    for person in subset.astype(int):
        #print(person)
        for i in range(0,18):
            node_index = person[i]
            #print(node_index)
            x1,y1 = candidate[node_index][:2]
            #print(x1,y1)
        #print("candidate"+str(candidate))
        #print(subset)
            all_pose_peaks.append((x1,y1))
        #print(all_pose_peaks)
    for pose_index, (x, y) in enumerate(all_pose_peaks):
        cv2.circle(oriImg, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(oriImg, str(pose_index), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)  # 在点的上方显示索引
    return [oriImg,all_pose_peaks]
if __name__ == "__main__":

    image = cv2.imread("build_model/angelbeats_image/kanade.png")
    info = detect_pose(image)
    body_pose_information = info[1].copy()

    head_width = math.sqrt(abs(body_pose_information[17][0] - body_pose_information[16][0])**2 + abs(body_pose_information[17][1] - body_pose_information[16][1])**2)
    body_width = math.sqrt(abs(body_pose_information[11][0] - body_pose_information[8][0])**2 + abs(body_pose_information[11][1] - body_pose_information[8][1])**2)
    body_height = math.sqrt(abs(body_pose_information[1][0] - body_pose_information[8][0])**2 + abs(body_pose_information[1][1] - body_pose_information[8][1])**2)
    shoulder = (math.sqrt(abs(body_pose_information[5][0] - body_pose_information[2][0])**2 + abs(body_pose_information[5][1] - body_pose_information[2][1])**2) - body_width)/2
    
    left_arm_1 = math.sqrt(abs(body_pose_information[2][0] - body_pose_information[3][0])**2 + abs(body_pose_information[2][1] - body_pose_information[3][1])**2)
    right_arm_1 = math.sqrt(abs(body_pose_information[5][0] - body_pose_information[6][0])**2 + abs(body_pose_information[5][1] - body_pose_information[6][1])**2)

    left_arm_2 = math.sqrt(abs(body_pose_information[3][0] - body_pose_information[4][0])**2 + abs(body_pose_information[3][1] - body_pose_information[4][1])**2)
    right_arm_2 = math.sqrt(abs(body_pose_information[6][0] - body_pose_information[7][0])**2 + abs(body_pose_information[6][1] - body_pose_information[7][1])**2)

    left_leg_1 = math.sqrt(abs(body_pose_information[9][0] - body_pose_information[8][0])**2 + abs(body_pose_information[9][1] - body_pose_information[8][1])**2)
    right_leg_1 = math.sqrt(abs(body_pose_information[11][0] - body_pose_information[12][0])**2 + abs(body_pose_information[11][1] - body_pose_information[12][1])**2)

    left_leg_2 = math.sqrt(abs(body_pose_information[9][0] - body_pose_information[10][0])**2 + abs(body_pose_information[9][1] - body_pose_information[10][1])**2)
    right_leg_2 = math.sqrt(abs(body_pose_information[12][0] - body_pose_information[13][0])**2 + abs(body_pose_information[12][1] - body_pose_information[13][1])**2)

    print("info_1:",head_width,body_width,body_height,shoulder)
    print("info_2:",left_arm_1,right_arm_1,left_arm_2,right_arm_2)
    print("info_3:",left_leg_1,right_leg_1,left_leg_2,right_leg_2)

    arm_1 = (left_arm_1 + right_arm_1) / 2
    arm_2 = (left_arm_2 + right_arm_2) / 2
    leg_1 = (left_leg_1 + right_leg_1) / 2
    leg_2 = (left_leg_2 + right_leg_2) / 2

    print("head:",head_width)
    print("body_width:",body_width,"body_height:",body_height)
    print("shoulder:",shoulder)
    print("arm_info:",arm_1,arm_2)
    print("leg_info:",leg_1,leg_2)
    
    body_width_ratio = body_width / head_width
    body_height_ratio = body_height / head_width
    shoulder_ratio = shoulder / head_width
    arm_1_ratio = arm_1 / head_width
    arm_2_ratio = arm_2 / head_width
    leg_1_ratio = leg_1 / head_width
    leg_2_ratio = leg_2 / head_width

    print("head (作为单位1):", 1)
    print("body_width_ratio:", body_width_ratio,"body_height_ratio:",body_height_ratio)
    print("shoulder_ratio:", shoulder_ratio)
    print("arm_info_ratio:", arm_1_ratio, arm_2_ratio)
    print("leg_info_ratio:", leg_1_ratio, leg_2_ratio)


    character_data = {
    "character":[
        {"name":"Kanade","head_width":head_width,"body_width_ratio":body_width_ratio,
         "body_height_ratio":body_height_ratio,"shoulder_ratio":shoulder_ratio,
         "arm_info_ratio":[arm_1_ratio, arm_2_ratio],"leg_info_ratio":[leg_1_ratio, leg_2_ratio]}
    ]
    }

    with open('character_data.json', 'w', encoding='utf-8') as f:
        json.dump(character_data, f, ensure_ascii=False, indent=4)

    json_data = json.dumps(character_data)

    #cv2.imshow("image",info[0])
    #cv2.waitKey(0)

    # 创建 socket 服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 65432))  # 绑定到本地主机和端口
    server_socket.listen(1)  # 监听传入的连接

    print('等待连接...')
    conn, addr = server_socket.accept()
    print('连接到:', addr)

    # 发送 JSON 数据
    conn.sendall(json_data.encode('utf-8'))
    # 关闭连接
    conn.close()





    