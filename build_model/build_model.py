import bpy
import time,math
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
        #创建一个骨骼系统
        bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(self.body_width/4,0,self.leg_info[1]+self.leg_info[0]-self.body_height/2))
        armature = bpy.context.object  #获取创建的骨骼系统对象
        armature.name = 'Character_Armature'  #给骨骼系统命名
        
        #进入编辑模式
        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = armature.data.edit_bones  #获取编辑模式下的骨骼数据

        #修改默认骨骼
        bone = edit_bones.active  # 获取当前活跃的骨骼（默认创建的那个）
        bone.name = 'MainBone'  # 给骨骼命名
        bone.head = (0,0,0)  # 设置骨骼头部的位置（局部坐标）
        bone.tail = (0,0,self.body_height)  # 设置骨骼尾部的位置，向X轴方向延伸1m

        armature.select_set(True)
        bpy.ops.armature.select_all(action='SELECT')
        bpy.ops.armature.subdivide(number_cuts=3)#细分


        body = bpy.data.objects.get("body")
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 切换到顶点选择模式
        bpy.ops.mesh.select_mode(type="VERT")  
        # 执行细分操作，将对象分为四段
        bpy.ops.mesh.subdivide(number_cuts=3)
        
        
        
        # # 返回到对象模式
        bpy.ops.object.mode_set(mode='OBJECT')

        # 首先取消选择所有对象
        bpy.ops.object.select_all(action='DESELECT')
        # 选择名为 "body" 的对象
        bpy.ops.object.select_pattern(pattern="body")
        # 选择名为 "Character_Armature" 的对象
        bpy.ops.object.select_pattern(pattern="Character_Armature")

        # 可选：设置其中一个对象为活动对象
        bpy.context.view_layer.objects.active = bpy.data.objects['Character_Armature']
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        
        #create_armature('MyBone',(),(0,0,self.leg_info[1]/2),(0,0,0.2))

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