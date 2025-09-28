'''模型训练与测试
'''
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import time
import torch
import ultralytics
print(ultralytics.__version__)  # 查看当前 ultralytics 版本
def gpu_state():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA 可用")
        # 获取 PyTorch 安装的 CUDA 版本
        cuda_version = torch.version.cuda
        print(f"PyTorch 安装的 CUDA 版本为: {cuda_version}")
        # 获取可用的 CUDA 设备
        device = torch.device("cuda")
        # 创建一个张量
        x = torch.tensor([1.0, 2.0, 3.0])
        # 将张量移动到 GPU 上
        x = x.to(device)
        print(f"张量已成功移动到 {torch.cuda.get_device_name(device)} 上")
    else:
        print("CUDA 不可用，可能未正确安装支持 CUDA 的 PyTorch 版本")
        cuda_version = torch.version.cuda
        if cuda_version:
            print(f"当前 PyTorch 安装的 CUDA 版本为: {cuda_version}")
        else:
            print("当前 PyTorch 未安装 CUDA 支持")
def timer(func):
    """
    一个装饰器函数，用于计算函数执行的时间。
    
    Args:
    func (function): 需要计时的函数。
    
    Returns:
    function: 包装后的函数，能够计算执行时间。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算执行时间
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result
    return wrapper
def train_model():
    model = YOLO(r"D:\prj\ai\yolov10\code\yolov10n.pt")
    result = model.train(data="D:/ai_dataset/fly/anti-uav/3rd_Anti-UAV_train_val/dataset.yaml",epochs=100,imgsz=640)
    print(result)
    # 模型输入所需的图像大小。可以是正方形图像的整数（例如， 640 对于 640x640）或元组 (height, width) 用于指定特定维度。
    model.export(format="onnx",imgsz=(512,640),dynamic=True)  # 导出为 ONNX 格式，动态批处理大小


def train_test():
    model = YOLO(r"D:\prj\ai\yolov10\runs\detect\train\weights\best.onnx")

    train_source_dir = r"D:\ai_dataset\fly\anti-uav\3rd_Anti-UAV_train_val\track1_test\20190925_101846_1_4"
    train_files = [f for f in os.listdir(train_source_dir) if f.endswith('.jpg')]
    train_files.sort()
        
    for img_file in train_files:
        src_img = os.path.join(train_source_dir, img_file)
        results = model.predict(src_img)
        # 获取预测结果中的图像（已经绘制了边界框等）
        annotated_image = results[0].plot()  # 这会返回带有标注的图像数组
        # 使用OpenCV显示图像
        cv2.imshow("Prediction Result", annotated_image)
        cv2.waitKey(1)  # 等待按键

    cv2.destroyAllWindows()

if __name__ == '__main__':
    gpu_state()
    #train_model()
    #train_test()