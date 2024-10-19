import bpy
import mathutils,math
import json,socket


def create_basic_cube(head_width,cube_name,location,x,y,z):
    #print(head_width)
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    cube = bpy.context.object
    cube.scale.x = x
    cube.scale.y = y
    cube.scale.z = z
    cube.name = cube_name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
def create_basic_circle(circle_name,location,radius):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,          # 球体的半径
        location=location,      # 球体的中心位置
        segments=16,      # 经度细分（水平切割数）
        ring_count=16        # 纬度细分（垂直切割数）
    )
    circle = bpy.context.object
    circle.name = circle_name

def Convert_Pose(armature_matrix_world,target_pose,target_object):
    target_pose[0] = mathutils.Vector(target_pose[0])
    target_pose[1] = mathutils.Vector(target_pose[1])

    target_head_position = armature_matrix_world.inverted() @ target_pose[0]
    target_tail_position = armature_matrix_world.inverted() @ target_pose[1]

    target_object.head = target_head_position
    target_object.tail = target_tail_position

class character_model:
    def __init__(self,name,head_width,body_width_ratio,body_height_ratio,shoulder_ratio,arm_info_ratio,leg_info_ratio):

        print("角色名:",name)
        print(head_width,body_width_ratio,body_height_ratio,shoulder_ratio,arm_info_ratio,leg_info_ratio)

        self.name = name
        self.head_width = 0.2
        self.body_width = body_width_ratio * self.head_width
        self.body_height = body_height_ratio * self.head_width
        self.shoulder = shoulder_ratio * self.head_width
        self.arm_info = [arm_info_ratio[0] * self.head_width,arm_info_ratio[1] * self.head_width]
        self.leg_info = [leg_info_ratio[0] * self.head_width,leg_info_ratio[1] * self.head_width]

    def build_basic_model(self):
        create_basic_cube(self.head_width,"left_leg_2",(0,0,0),0.05,0.06,self.leg_info[1])
        
        create_basic_cube(self.head_width,"right_leg_2",(self.body_width/2,0,0),
                          0.05,0.06,self.leg_info[1])
        
        create_basic_cube(self.head_width,"left_leg_1",(0,0,self.leg_info[1]),
                          0.065,0.06,self.leg_info[0])
        
        create_basic_cube(self.head_width,"right_leg_1",(self.body_width/2,0,self.leg_info[1]),
                          0.065,0.06,self.leg_info[0])
        
        create_basic_cube(self.head_width,"body",(self.body_width/4,0,self.leg_info[1]+self.leg_info[0]),
                           self.body_width,0.12,self.body_height)
        
        create_basic_circle("shoulder1",(self.body_width/2*-1,0,self.leg_info[1]+self.leg_info[0]+self.body_height/2),self.shoulder/2)
        
        create_basic_circle("shoulder2",(self.body_width,0,self.leg_info[1]+self.leg_info[0]+self.body_height/2),self.shoulder/2)
        
        create_basic_cube(self.head_width,"left_arm_1",(self.body_width/2*-1,0,self.leg_info[1]+self.leg_info[0]+self.body_width-self.arm_info[0]/3),0.05,0.06,self.arm_info[0])
        
        create_basic_cube(self.head_width,"right_arm_1",(self.body_width,0,self.leg_info[1]+self.leg_info[0]+self.body_width-self.arm_info[0]/3),0.05,0.06,self.arm_info[0])
        
        create_basic_cube(self.head_width,"left_arm_2",(self.body_width/2*-1,0,self.leg_info[1]+self.leg_info[0]+self.body_width-self.arm_info[0]/3-(self.arm_info[1]+(abs(self.arm_info[0]-self.arm_info[1]))/2)),0.05,0.06,self.arm_info[1])
        
        create_basic_cube(self.head_width,"right_arm_2",(self.body_width,0,self.leg_info[1]+self.leg_info[0]+self.body_width-self.arm_info[0]/3-(self.arm_info[1]+(abs(self.arm_info[0]-self.arm_info[1]))/2)),0.05,0.06,self.arm_info[1])
        
        create_basic_circle("head",(self.body_width/4,0,self.leg_info[1]+self.leg_info[0]+self.body_height/2+self.head_width/2),self.head_width/2)


        bpy.ops.object.select_all(action='DESELECT')
        # 定义要合并的对象名称列表
        object_names = [ "head","shoulder1","shoulder2","left_arm_1",
                        "right_arm_1","left_arm_2","right_arm_2","left_leg_2",
                        "right_leg_2","left_leg_1","right_leg_1","body"]
        # 选择并设置活动对象
        for name in object_names:
            obj = bpy.data.objects.get(name)
            if obj:
                bpy.ops.object.select_pattern(pattern=name)
                bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()


    def build_armature(self):
        # 添加一个 basic human 元骨架
        bpy.ops.object.armature_basic_human_metarig_add()
        rig = bpy.context.view_layer.objects.active
        rig.location = (self.body_width/4, 0, -self.leg_info[1]/2)
        # 进入编辑模式
        bpy.ops.object.mode_set(mode='EDIT')

        # 获取当前活跃对象（假设它是刚刚添加的元骨架）
        armature = bpy.context.active_object.data
        edit_bones = armature.edit_bones

        # 删除不需要的骨骼，例如删除头部骨骼
        bones_to_remove = ['pelvis.R','pelvis.L','breast.R','breast.L','toe.L',
                        'toe.R','heel.02.L','heel.02.R','foot.L','foot.R','spine.002','spine.003','spine.004','spine.005']
        #bones_to_remove = ['pelvis.R','pelvis.L','breast.R','breast.L']
        for bone_name in bones_to_remove:
            if bone_name in edit_bones:
                edit_bones.remove(edit_bones[bone_name])



        bpy.ops.object.mode_set(mode='EDIT')
        armature = bpy.data.objects.get("metarig")
        # 获取Rigify骨架的编辑骨骼数据
        edit_bones = armature.data.edit_bones
        armature = bpy.context.active_object.data
        edit_bones = armature.edit_bones

        armature = bpy.context.view_layer.objects.active
        armature_matrix_world = armature.matrix_world

        body_part_pose = {"spine":[(self.body_width/4,0,self.leg_info[1]+self.leg_info[0]-self.body_height/2),(self.body_width/4,0,self.leg_info[1]+self.leg_info[0]+self.body_height/2)],
                          "thigh.R":[],
                          ""}
        for part_name,part_pose in body_part_pose.items():
            print(part_name,part_pose)
            Convert_Pose(armature_matrix_world,part_pose,edit_bones[part_name])


        bpy.ops.object.mode_set(mode='OBJECT')    



client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 65432))

received_data = client_socket.recv(1024)

json_data = received_data.decode('utf-8')

data = json.loads(json_data)
print('receive:', data)

character_info = data["character"][0]

character = character_model(
    name=character_info["name"],
    head_width=character_info["head_width"],
    body_width_ratio=character_info["body_width_ratio"],
    body_height_ratio=character_info["body_height_ratio"],
    shoulder_ratio=character_info["shoulder_ratio"],
    arm_info_ratio=character_info["arm_info_ratio"],
    leg_info_ratio=character_info["leg_info_ratio"]
)

character.build_basic_model()
character.build_armature()

print("The End")