import bpy
import mathutils
import math,time

bpy.ops.object.mode_set(mode='EDIT')
armature = bpy.data.objects.get("metarig")
# 获取Rigify骨架的编辑骨骼数据
edit_bones = armature.data.edit_bones


 # 获取骨架对象的世界矩阵
armature_matrix_world = armature.matrix_world


# 你可以重复上面的步骤调整其他骨骼
# 例如调整spine的头部和尾部位置：
target_spine_head = (0.0, 0.0, 0.9)  # 假设spine的起始位置
target_spine_tail = (0.0, 0.0, 1.2)  # 假设spine的结束位置
def Convert_Pose(target_pose,target_object):
    target_pose = mathutils.Vector(target_pose)

    target_head_position = armature_matrix_world.inverted() @ target_pose[0]
    target_tail_position = armature_matrix_world.inverted() @ target_pose[1]

    target_object.head = target_head_position
    target_object.tail = target_tail_position

if True:
    armature = bpy.context.active_object.data
    edit_bones = armature.edit_bones
    edit_bones['spine'].head = target_spine_head
    edit_bones['spine'].tail = target_spine_tail

    # 获取骨骼头部和尾部的局部位置
    local_head_position = edit_bones['spine'].head
    local_tail_position = edit_bones['spine'].tail

    # 计算骨骼的全局位置
    global_head_position = armature_matrix_world @ local_head_position
    global_tail_position = armature_matrix_world @ local_tail_position
    Convert_Pose([target_spine_head,target_spine_tail],edit_bones['spine'])
# 调整完成后，切换回对象模式
bpy.ops.object.mode_set(mode='OBJECT')