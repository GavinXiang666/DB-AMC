import bpy
bpy.ops.wm.append(directory="C:/Users/gavin/pytorch-openpose-blender/openpose_template.blend/Object",filename="Armature_Old")
def apply_pose(action):
    area = [area for window in bpy.context.window_manager.windows for area in window.screen.areas if area.ui_type == 'ASSETS']
    
    bpy.ops.pose.transforms_clear()
    
    w = bpy.context.window
    
    for a in area:
        for s in a.spaces:
            if s.type == 'FILE_BROWSER':
                space = s

armature = bpy.data.objects.get("Armature_Old")
        
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='EDIT')
edit_bones = armature.data.edit_bones
        
armature = bpy.context.view_layer.objects.active
armature_matrix_world = armature.matrix_world

#暂时禁用骨骼父子关系
bone_parent_dict = {}
bone_constraints_dict = {}

armature = bpy.data.objects.get('Armature_Old')
if armature:
    bpy.context.view_layer.objects.active = armature

    # 进入编辑模式，禁用父子关系
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in armature.data.edit_bones:
        if bone.parent:
            bone_parent_dict[bone.name] = bone.parent.name
            bone.parent = None
    
    # 禁用X轴镜像
    armature.data.use_mirror_x = False