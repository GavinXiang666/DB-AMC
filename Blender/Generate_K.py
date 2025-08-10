import bpy
import numpy as np
import json
import os

def setup_rig(rig_name):
    bpy.ops.object.mode_set(mode='OBJECT')
    rig = bpy.data.objects.get(rig_name)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    return rig

"""归一化向量"""
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

if __name__ == "__main__":
    '''初始化rig对象'''
    Rig = setup_rig("CT")
    bpy.ops.object.mode_set(mode='POSE')

    k_data_list = []
    for k,bone_name in enumerate(['torso','head','upper_arm_fk.L','forearm_fk.L','upper_arm_fk.R',
                    'forearm_fk.R','thigh_fk.L','shin_fk.L','thigh_fk.R','shin_fk.R']):
        pose_bone = Rig.pose.bones[bone_name]
        #获取物体的局部转换矩阵
        matrix = pose_bone.matrix
        #局部Y轴
        ky = np.array(matrix.col[1].xyz)
        #局部Z轴
        kz = np.array(matrix.col[2].xyz)
        #叉乘计算kx
        kx = normalize(np.cross(ky,kz))

        k_data = {
            'bone_name':bone_name,
            'kx':kx.tolist(),
            'ky':ky.tolist(),
            'kz':kz.tolist()
        }#需转化为列表

        k_data_list.append(k_data)

    with open('D:/DB-AMC/Blender/K_Data_CT.json', 'w', encoding='utf-8') as f:
        json.dump(k_data_list, f, ensure_ascii=False, indent=4)

