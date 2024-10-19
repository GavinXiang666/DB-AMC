import bpy
import math
import csv

def rotate_bone(armature, bone_name, rotate_angle_list, body=False):
    # 将角度从字符串转换为浮点数，并转换为弧度制
    rotate_angle_list = [float(item) for item in rotate_angle_list]
    rotate_angle_list = [math.radians(angle) for angle in rotate_angle_list]
    
    # 进入姿态模式
    bpy.ops.object.mode_set(mode='POSE')
    bone = armature.pose.bones[bone_name]
    
    bone.rotation_mode = 'XYZ'  # 设置为 XYZ 欧拉旋转模式
    
    # 设置旋转值（相对于上一帧）
    if body:
        # 针对身体骨骼的旋转，依次应用 X、Y、Z 轴上的旋转
        bone.rotation_euler[0] += rotate_angle_list[1]  # X 轴旋转
        bone.rotation_euler[1] += rotate_angle_list[2]  # Y 轴旋转
        bone.rotation_euler[2] += rotate_angle_list[0]  # Z 轴旋转
    else:
        # 针对四肢骨骼的旋转，通常只使用 X 和 Z 轴
        bone.rotation_euler[0] += rotate_angle_list[1]  # X 轴旋转
        bone.rotation_euler[1] += 0                     # 忽略 Y 轴旋转
        bone.rotation_euler[2] += rotate_angle_list[0]  # Z 轴旋转

def insert_keyframe(armature, bone_name, frame):
    # 获取骨骼并为其rotation_euler属性插入关键帧
    bone = armature.pose.bones[bone_name]
    bone.keyframe_insert(data_path="rotation_euler", frame=frame)

if __name__ == "__main__":
    # 获取 Armature_Old 对象
    armature = bpy.data.objects['Armature_Old']
    
    # 切换到对象模式以确保可以进入姿态模式
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 选择 Armature_Old
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    # 初始帧为1
    current_frame = 1
    frames_per_second = 8  # 每秒8帧
    
    # 打开CSV文件并逐行读取
    with open('C:/Users/gavin/DB-AMC/AMC/output/bone_rotate_angle.csv', 'r', newline='') as angle_file:
        csvreader = csv.reader(angle_file)
        current_action_data = []  # 存储当前动作帧的骨骼数据
        
        # 遍历CSV中的每一行数据
        for line_index, line in enumerate(csvreader):
            # 骨骼名称转换
            if line[0] == "0":
                line[0] = "Head"
            elif line[0] == "1":
                line[0] = "Spine"
            elif line[0] == "2":
                line[0] = "Arm_L"
            elif line[0] == "3":
                line[0] = "Forearm_L"
            elif line[0] == "5":
                line[0] = "Arm_R"
            elif line[0] == "6":
                line[0] = "Forearm_R"
            elif line[0] == "8":
                line[0] = "Thigh_L"
            elif line[0] == "9":
                line[0] = "Shin_L"
            elif line[0] == "11":
                line[0] = "Thigh_R"
            elif line[0] == "12":
                line[0] = "Shin_R"
            
            # 读取并解析旋转角度
            angle_string = line[1]
            angle_values = angle_string.strip("[]").split(", ")
            
            # 如果读取了9个骨骼数据，就创建一个新动画帧
            if len(current_action_data) == 9:
                current_frame += frames_per_second
                for bone_data in current_action_data:
                    bone_name = bone_data[0]
                    insert_keyframe(armature, bone_name, current_frame)
                current_action_data.clear()  # 清空当前动作帧的数据
            
            # 旋转骨骼
            if len(angle_values) == 3:
                rotate_bone(armature, line[0], angle_values, body=True)
            else:
                rotate_bone(armature, line[0], angle_values)
            
            # 将此骨骼的数据添加到当前动作帧
            current_action_data.append(line)
        
        # 处理最后剩下的骨骼数据
        if current_action_data:
            current_frame += frames_per_second
            for bone_data in current_action_data:
                bone_name = bone_data[0]
                insert_keyframe(armature, bone_name, current_frame)
