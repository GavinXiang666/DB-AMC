# DB-AMC(DreamBusters! Automatic Motion Capture)

![](https://github.com/GavinXiang666/DB-AMC/blob/main/logo/logo_black.jpg)

基于yolov7实现的简易摄像头动捕装置，将捕获到的信息用sqlite3存储并在Blender上用脚本做映射

## 开发环境

- Windows11
- Python 3.10.16(Anaconda虚拟环境)
- Blender 4.5.1 LTS(启用自带Rigify插件)

## 运行项目
##### 在项目目录下运行

```bash
pip3 install -r requirements.txt
```
##### 若下载失败可尝试换源

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple #清华源
```
##### 下载模型文件
在 https://github.com/WongKinYiu/yolov7 下载`yolov7-w6-pose.pt`文件并放在`AMC/model`目录下