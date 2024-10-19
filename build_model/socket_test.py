import socket
import json

# 要发送的数据
data = {
    'name': 'Blender',
    'version': '3.4',
    'features': ['modeling', 'animation', 'rendering']
}

# 将数据转换为 JSON 字符串
json_data = json.dumps(data)

# 创建 socket 服务器
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 65432))  # 绑定到本地主机和端口
server_socket.listen(1)  # 监听传入的连接

print('等待连接...')
conn, addr = server_socket.accept()
print('连接到:', addr)

# 发送 JSON 数据
conn.sendall(json_data.encode('utf-8'))

# 关闭连接
conn.close()
