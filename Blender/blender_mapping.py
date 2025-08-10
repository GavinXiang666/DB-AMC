import bpy
import sqlite3
import os
import numpy as np
from math import acos, atan2, sin, cos, sqrt
import json

'''
连接sqlite数据库
'''
conn = sqlite3.connect("D:/DB-AMC/sql/pose_database.db")
cursor = conn.cursor()


def setup_rig(rig_name):
    bpy.ops.object.mode_set(mode='OBJECT')
    rig = bpy.data.objects.get(rig_name)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    return rig

def rotate_bone(rig,bone_name,rotation_list):
    rig.pose.bones[bone_name].rotation_euler[0] += np.radians(rotation_list[0])
    rig.pose.bones[bone_name].rotation_euler[1] += np.radians(rotation_list[1])

"""归一化向量"""
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

'''投影向量计算'''
def vector_to_plane_projection(v, n):

    dot_product = np.dot(v, n)
    
    n_mag_sq = np.dot(n, n)
    
    v_perpendicular = (dot_product / n_mag_sq) * n
    
    v_parallel = v - v_perpendicular
    
    return v_parallel
'''计算向量旋转'''
def vector_rotation(v1,v2,kx,ky):
    plane_vector = np.cross(v1,kx)
    #法向量
    norm = np.cross(v1,plane_vector)
    norm_vector = normalize(norm)
    middle_vector = vector_to_plane_projection(v2,norm_vector)
    
    vector_f = np.cross(v1,middle_vector)
    theta1 = np.arccos((v1@middle_vector)/(np.linalg.norm(v1)*np.linalg.norm(middle_vector)))
     #用叉乘向量与点积判断旋转角度是否为正
    if vector_f @ kx >= 0:
        theta1 = theta1
    elif vector_f @ kx < 0:
        theta1 = -theta1
    
    vector_e = np.cross(middle_vector,v2)
    theta2 = np.arccos((v2@middle_vector)/(np.linalg.norm(v2)*np.linalg.norm(middle_vector)))
    #用叉乘向量与点积判断旋转角度是否为正
    if vector_e @ ky >= 0:
        theta2 = theta2
    elif vector_e @ ky < 0:
        theta2 = -theta2

    return np.degrees(theta1),np.degrees(theta2)

#更新向量坐标
def update_vector(pitch,yaw,roll,vector):
    pitch,yaw = np.radians(pitch),np.radians(yaw)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Rx @ Ry @ Rz#构造旋转矩阵

    v_rotated = R @ vector
    return v_rotated


def rotate_k(v, u, theta_deg):
    #转换为numpy数组
    v = np.array(v, dtype=np.float64)
    u = np.array(u, dtype=np.float64)
    
    #检查旋转轴是否为零向量
    if np.linalg.norm(u) < 1e-10:
        raise ValueError("旋转轴不能是零向量")
    
    #归一化旋转轴
    u_hat = u / np.linalg.norm(u)
    
    #转换角度为弧度
    theta = np.radians(theta_deg)
    
    #计算点积和叉积
    dot_product = np.dot(u_hat, v)
    cross_product = np.cross(u_hat, v)
    
    #应用罗德里格斯公式
    v_rot = (v * np.cos(theta) 
             + cross_product * np.sin(theta) 
             + u_hat * dot_product * (1 - np.cos(theta)))
    
    return v_rot

def json_reader(file_path,bone_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        k_data = json.load(f)
        for data in k_data:
            if data['bone_name'] == bone_name:
                return data 

    
'''骨骼类'''
class Armature: 
    def __init__(self,pose,rig,first_state=False):
        self.pose = pose
        self.rig = rig#需要在Blender中操作的对象
        self.first_state = first_state#是否为初始状态骨骼，默认False

        def get_keypoint(keypoint_id):
            return np.array([self.pose[keypoint_id][1],
                             self.pose[keypoint_id][2],self.pose[keypoint_id][3]])
        
        left_ear = get_keypoint(3)
        right_ear = get_keypoint(4)
        left_shoulder = get_keypoint(5)
        right_shoulder = get_keypoint(6)
        left_elbow = get_keypoint(7)
        right_elbow = get_keypoint(8)
        left_wrist = get_keypoint(9)
        right_wrist = get_keypoint(10)
        left_hip = get_keypoint(11)
        right_hip = get_keypoint(12)
        left_knee = get_keypoint(13)
        right_knee = get_keypoint(14)
        left_ankle = get_keypoint(15)
        right_ankle = get_keypoint(16)

        self.blender_pose = {
            "torso":(left_shoulder+right_shoulder)/2 - (left_hip+right_hip)/2,
            "head":(left_ear+right_ear)/2 - (left_shoulder+right_shoulder)/2,
            "upper_arm_fk.L":left_elbow - left_shoulder,
            "upper_arm_fk.R":right_elbow - right_shoulder,
            "forearm_fk.L":left_wrist - left_elbow,
            "forearm_fk.R":right_wrist - right_elbow,
            "thigh_fk.L":left_knee - left_hip,
            "thigh_fk.R":right_knee - right_hip,
            "shin_fk.L":left_ankle - left_knee,
            "shin_fk.R":right_ankle - right_knee
        }

        self.xyz = {}
        
        #首个骨骼对象需要手动记录下每个骨骼的旋转轴坐标
        if self.first_state == True:
            for n,bone_name in enumerate(['torso','head','upper_arm_fk.L','forearm_fk.L','upper_arm_fk.R',
                       'forearm_fk.R','thigh_fk.L','shin_fk.L','thigh_fk.R','shin_fk.R']):
                
                # #Torso特殊处理
                # if bone_name != 'torso':
                #     ky = normalize(self.blender_pose[bone_name])
                # else:
                #     ky = np.array([1,0,0])


                # #手动定位kz
                # if bone_name in ['shin_fk.R','shin_fk.L','thigh_fk.R','thigh_fk.L']:
                #     kz = np.array([-1,0,0])
                # elif bone_name == "torso":
                #     kz = np.array([0,0,1])
                # else:
                #     kz = np.array([1,0,0])
                
                # #叉乘计算kx
                # kx = normalize(np.cross(ky,kz))
                
                k_data = json_reader('D:/DB-AMC/Blender/K_Data_CT.json',bone_name)
                kx = np.array(k_data['kx'])
                ky = np.array(k_data['ky'])
                kz = np.array(k_data['kz'])
                self.xyz[bone_name] = {
                    'kx':kx,
                    'ky':ky,
                    'kz':kz
                }
                
    def rotate(self,target_armature):
        def update_pose(pitch,yaw,control_bone):
            with open('D:/DB-AMC/Blender/paternity.json', 'r') as file: 
                paternity = json.load(file)
                file.close()
        
            #LinkedBones表示该点还关联了哪些点，这个点的运动会让关联点一起运动
            LinkedBones = []
            for bone in paternity:
                if bone["id"] == control_bone:
                    LinkedBones = bone["LinkedBones"]
                    break
            kx_control = self.xyz[control_bone]['kx']
            ky_control = self.xyz[control_bone]['ky']
            kz_control = self.xyz[control_bone]['kz']

            if LinkedBones != []:
                for single_bone in LinkedBones:

                    self.blender_pose[single_bone] = rotate_k(self.blender_pose[single_bone],kx_control,pitch)
                    self.blender_pose[single_bone] = rotate_k(self.blender_pose[single_bone],ky_control,yaw)
                    '''更新旋转轴'''
                    if single_bone == control_bone:#如果旋转自身
                        pass
                    else:
                        #X
                        self.xyz[bone_name]['kx'] = rotate_k(self.xyz[bone_name]['kx'],kx_control,pitch)
                        self.xyz[bone_name]['kx'] = rotate_k(self.xyz[bone_name]['kx'],ky_control,yaw)
                        #Y
                        self.xyz[bone_name]['ky'] = rotate_k(self.xyz[bone_name]['ky'],kx_control,pitch)
                        self.xyz[bone_name]['ky'] = rotate_k(self.xyz[bone_name]['ky'],ky_control,yaw)
                        #Z
                        self.xyz[bone_name]['kz'] = rotate_k(self.xyz[bone_name]['kz'],kx_control,pitch)
                        self.xyz[bone_name]['kz'] = rotate_k(self.xyz[bone_name]['kz'],ky_control,yaw)
                        
                        


        for k,bone_name in enumerate(['torso','head','upper_arm_fk.L','forearm_fk.L','upper_arm_fk.R',
                       'forearm_fk.R','thigh_fk.L','shin_fk.L','thigh_fk.R','shin_fk.R']):
            # bpy.ops.object.mode_set(mode='OBJECT')
            # rig = bpy.data.objects.get('rig')

            # pose_bone = rig.pose.bones[bone_name]
            # #获取物体的局部转换矩阵
            # matrix = pose_bone.matrix
            # #获取局部x,y,z轴方向向量
            # kx = np.array(matrix.col[0].xyz) #局部X轴
            # ky = np.array(matrix.col[1].xyz) #局部Y轴
            # kz = np.array(matrix.col[2].xyz) #局部Z轴


            kx = self.xyz[bone_name]['kx']
            ky = self.xyz[bone_name]['ky']
            kz = self.xyz[bone_name]['kz']

            if bone_name == "forearm_fk.R":
                print('AAAAA',self.blender_pose[bone_name],target_armature.blender_pose[bone_name],kx,ky)

            pitch,yaw = vector_rotation(self.blender_pose[bone_name],target_armature.blender_pose[bone_name],kx,ky)

            bpy.ops.object.mode_set(mode='POSE')
            rotate_bone(self.rig,bone_name,[pitch,yaw])
            update_pose(pitch,yaw,control_bone=bone_name)
        
if __name__ == "__main__":
    '''初始化rig对象'''
    angel_rig = setup_rig("CT")
    bpy.ops.object.mode_set(mode='POSE')

    '''读取数据库的坐标信息'''
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    pose_list = []
    for table in tables:
        pose_info = []
        cursor.execute(f"SELECT * FROM {table[0]};")
        rows = cursor.fetchall()
        for row in rows:
            pose_info.append(row)
        pose_list.append(pose_info)

    armature_object_list = []
    for i,armature in enumerate(pose_list):
        if i == 0:
            new_armature = Armature(armature,angel_rig,first_state=True)
        else:
            new_armature = Armature(armature,angel_rig)
        #添加对象
        armature_object_list.append(new_armature)
    
    initial_armature_object = armature_object_list[0]
    target_armature_object = armature_object_list[1]
    
    initial_armature_object.rotate(target_armature_object)
    