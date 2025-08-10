
import torch
import cv2
from torchvision import transforms
import numpy as np

import os,sys
# 获取当前脚本所在目录（AMC/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 yolov7-main 目录路径
yolov7_dir = os.path.join(current_dir, "yolov7-main")
# 添加到 sys.path
sys.path.append(yolov7_dir)

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

"""基本设置"""
device = torch.device("cpu")
weigths = torch.load('AMC/model/yolov7-w6-pose.pt', map_location=device,weights_only=False)
model = weigths['model']
_ = model.float().eval()

"""图片处理"""
image = cv2.imread('AMC/yolov7-main/zimiao_test_camera.jpg')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
#推理
output, _ = model(image)

"""输出推理结果"""
output = non_max_suppression_kpt(output, 0.2, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
with torch.no_grad():
    output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

for idx in range(output.shape[0]):
    keypoints_matrix = output[idx, 7:].T
    plot_skeleton_kpts(nimg,keypoints_matrix, 3)

    print(keypoints_matrix)
    #打包keypoints信息矩阵
    keypoints_dict = {
        0: {'name': 'nose', 'data': []},
        1: {'name': 'left_eye', 'data': []},
        2: {'name': 'right_eye', 'data': []},
        3: {'name': 'left_ear', 'data': []},
        4: {'name': 'right_ear', 'data': []},
        5: {'name': 'left_shoulder', 'data': []},
        6: {'name': 'right_shoulder', 'data': []},
        7: {'name': 'left_elbow', 'data': []},
        8: {'name': 'right_elbow', 'data': []},
        9: {'name': 'left_wrist', 'data': []},
        10: {'name': 'right_wrist', 'data': []},
        11: {'name': 'left_hip', 'data': []},
        12: {'name': 'right_hip', 'data': []},
        13: {'name': 'left_knee', 'data': []},
        14: {'name': 'right_knee', 'data': []},
        15: {'name': 'left_ankle', 'data': []},
        16: {'name': 'right_ankle', 'data': []}
    }
    #判断关键点数据是否符合要求
    if len(keypoints_matrix) >= 51:
        single_keypoint_index = 0
        for i in range(0,len(keypoints_matrix)):
            if i % 3 == 0:
                keypoints_dict[single_keypoint_index]['data'] = [
                    int(keypoints_matrix[i]),
                    int(keypoints_matrix[i + 1]),
                    float(keypoints_matrix[i + 2])
                ]
                single_keypoint_index += 1
            else:
                continue
    print("#"*24)
    print(keypoints_dict)

#显示结果
result = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
