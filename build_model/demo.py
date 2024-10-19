import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'angelbeats_image/kanade.png'
oriImg = cv2.imread(test_image)  # B,G,R order

candidate, subset = body_estimation(oriImg)

canvas = copy.deepcopy(oriImg)


canvas = util.draw_bodypose(canvas, candidate, subset)
# detect hand
hands_list = util.handDetect(candidate, subset, oriImg)


all_hand_peaks = []
for x, y, w, is_left in hands_list:
    #cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    #   cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
    #     plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
    #     plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)
    
canvas = util.draw_handpose(canvas, all_hand_peaks)


#以下部分为新增部分

all_pose_peaks = []
for person in subset.astype(int):
    print(person)
    for i in range(0,18):
        node_index = person[i]
        print(node_index)
        x1,y1 = candidate[node_index][:2]
        print(x1,y1)
    #print("candidate"+str(candidate))
    #print(subset)
        all_pose_peaks.append((x1,y1))
    print(all_pose_peaks)
for pose_index, (x, y) in enumerate(all_pose_peaks):
    cv2.circle(oriImg, (int(x), int(y)), 10, (0, 0, 255), -1)
    cv2.putText(oriImg, str(pose_index), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)  # 在点的上方显示索引
cv2.imshow("oriImg",oriImg)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
