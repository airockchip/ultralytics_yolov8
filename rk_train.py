'''
    rk模型训练及模型导出，使用了rk的yolov8版本,https://github.com/tyjsnz/ultralytics_yolov8/tree/rk_yolov8  分支为：rk_yolov8
    需要安装rknn-toolkit2, pip install rknn-toolkit2进行模型转换，可参见：https://www.yuque.com/juzipi-u0gdg/vfd9yl/orm538p3cbyvi83r 
    模型训练后已经 在ubuntu20.04下测试通过，将训练后的onnx复制到ubuntu下，使用如下转换命令：

    python3 -m rknn.api.rknn_convert -t rk3568 -i ./model_config.yml -o ./

    可参见：https://www.yuque.com/juzipi-u0gdg/vfd9yl/xb42wbo0vg9ssgv0 
    ubuntu下到以下目录查看
        cd ~/software/rknn-toolkit2-2.0.0-beta0/rknn-toolkit2/examples/onnx/yolov8/model

    更改了激活函数：
        改激活函数流程如下：把ultralytics/nn/modules/conv.py中的Conv类进行修改，将default_act = nn.SiLU() # default activation改为
        default_act = nn.ReLU() # default activation
'''
from ultralytics import YOLO
import torch
import time

def gpu_state():
    import ultralytics

    print(f"yolo版本：{ultralytics.__version__}")
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

@timer
def train_data(model_path,data_yaml_path,epochs=100,prj_name='hhyc_cxf'):
    """训练数据集
    Args:
        model_path (str): 模型路径
        data_yaml_path (str): 数据集路径 : 'H:\prj\2024\10\ai\yolov10\demo\irdata.yaml'
        epochs (int): 训练轮数


    """
    #model_path = r'H:\prj\2024\10\ai\yolov10\yolov10n.pt'
    #modelpath = r'H:\prj\2024\10\ai\yolov10\demo\yolov10n.yaml'  # 会随机初始化参数
    # 从YAML构建并转移权重
    #model = YOLO(modelpath).load(model_path)  # load a pretrained model (recommended for training)


    # 加载预训练模型
    model = YOLO(model_path)

    cuda = 0 if torch.cuda.is_available() else "cpu"
    # Train the model
    """
        resume=True: YOLO 加载您指定的预训练模型，并在其基础上进行训练。
        freeze： 要冻结哪些层，例如[1, 2, 3]表示冻结第1、2、3层。
        lr0=0.01: 学习率初始值为0.01。


    """
    #freeze = [f'model.{x}.' for x in range(10)]  # 冻结前 5 层

    model.train(data=data_yaml_path,
                resume=False, # 是否从上次中断的训练状态继续训练
                epochs=epochs,
                project=prj_name,
                patience=30, # 表示在验证集性能没有提升的情况下，继续训练的轮数。如果在 patience 轮内验证集性能都没有改善，训练将停止。
                name='uav_yolov8n', # 结果保存的文件夹名称
                amp=False, # 是否使用自动混合精度训练。自动混合精度训练结合了单精度（FP32）和半精度（FP16）浮点数，在不显著损失模型精度的情况下，可以加快训练速度并减少内存使用。
                device=cuda,
                cache=True,
                exist_ok=True)
    
    
def train_val():
    import os
    import glob
    import cv2
    import torch

    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载预训练模型
    #model = YOLOv10(r"D:\prj\ai\yolov10\fly_ir\fly_ircar-yolov10n-0215\weights\best.pt").to(device)

    model = YOLO(r"D:\prj\ai\yolov8_rk\fly_model\uav_yolov8n\weights\best.pt").to(device)
    
    # model.val(batch=8,
    #           imgsz=640,
    #           data=r'E:\AI\ai\yolov10\flydata.yaml',
    #           save_dir=r'E:\AI\ai\fly\dataset\vvv')

    # 获取预测目录下所有图片的路径
    predict_dir = r"G:\prj\AI\ai-tracker\datasets\fly\images\val"
    imgs = glob.glob(os.path.join(predict_dir,'*.jpg'))
    
    for img in imgs:
        result = model.predict(img)
        results = result[0]
        #results[0].show()
        names   = results.names
        boxes   = results.boxes.data.tolist()

        img = cv2.imread(img)

        for obj in boxes:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            #color = random_color(label)
            

            cv2.rectangle(img, (left - 3, top - 33), (right, bottom), -1)

            # 绘制标签
            label = f"{int(label)}: {confidence:.2f}"
            cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
            img = cv2.resize(img,(1920,1080))
            cv2.imshow("a",img)
            cv2.waitKey(0)

    image_paths = [os.path.join(predict_dir, img) for img in os.listdir(predict_dir) if
                img.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 对每张图片进行推理
    # for image_path in image_paths:
    #     results = model.predict(image_path)
    
    #     # 显示预测结果
    #     results[0].show()


def model_export(model_path):
    model = YOLO(model_path)
    model.export(format="onnx")  # 导出为 ONNX 格式
    model.export(format="rknn")

if __name__ == '__main__':
    #copy_matching_files()

    #rename_wechat_images()

    gpu_state()
    pretrained_model_path = r"D:\prj\ai\yolov8_rk\yolov8n.pt"
    data_yaml_path = r"D:\prj\ai\yolov8_rk\data_yaml\dataset.yaml"
    train_data(pretrained_model_path,data_yaml_path,epochs=50,prj_name='fly_model50')

    print("tran success~ :) :)")

    model_export(r"D:\prj\ai\yolov8_rk\fly_model50\uav_yolov8n\weights\best.pt")
    
    train_val()