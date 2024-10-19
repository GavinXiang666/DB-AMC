import bpy

# 设置立方体的位置和尺寸
width = 4.0                 # 立方体的宽度（沿X轴）
height = 2.0                # 立方体的高度（沿Y轴）
depth = 1.0                 # 立方体的深度（沿Z轴）

# 创建一个新的立方体
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0))

# 获取刚刚创建的对象
cube = bpy.context.active_object

# 设置立方体的尺寸
cube.scale.x = width / 2.0
cube.scale.y = height / 2.0
cube.scale.z = depth / 2.0
cube.name = "desk"

# 应用变换以确保立方体的尺寸在编辑模式下是正确的
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
