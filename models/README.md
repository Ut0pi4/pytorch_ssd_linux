=====软件环境=====
操作系统：Ubuntu 16.04
GPU：GTX TITAN X
版本：
Python 3.7.7
Pytorch 1.3.1
CUDA 9.0.176
CUDNN 7.6.2

=====文件说明=====
首先，需要把数据集和权重文件下载之后解压在同一个文件夹里。
下载链接：
1.网站下载：
2.直接下载：

-- data文件夹：包含数据集里的图像和对应的boundingbox信息

-- darknet19_hr_75.52_92.73.pth：Darknet在ImageNet按448大小训练10epochs的权重文件

-- model1.pkl：训练好的YOLOv1模型文件

-- model2.pkl：训练好的YOLOv2模型文件

-- env.yml：软件环境文件

-- dataload_yolo.py：处理和读取数据的文件

包括读取数据，数据增强，resize图像和框的函数

-- YOLOv1.py：算法文件

首先在终端输入 python
再输入
import YOLOv1
from YOLOv1 import*

1)如果需要查看训练过程，
输入 YOLOv1.train() 即可，则会开始训练随机初始化的模型。
注：训练收敛速度较慢。
2)如果需要查看预测结果，
输入 YOLOv1.prediction() 即可，则会自动生成3张测试集的图片以及其预测结果，
如果想查看其他图片的预测结果，输入 YOLOv1.prediction(randpic=True) 即可，则会在测试集上随机取3张图片并进行预测。

-- YOLOv2.py：算法文件

首先在终端输入 python
再输入
import YOLOv2
from YOLOv2 import*

1)如果需要查看训练过程，
输入 YOLOv2.train() 即可，则会开始训练随机初始化的模型。
注：训练收敛速度较慢。
2)如果需要查看预测结果，
输入 YOLOv2.prediction() 即可，则会自动生成3张测试集的图片以及其预测结果，
如果想查看其他图片的预测结果，输入 YOLOv2.prediction(randpic=True) 即可，则会在测试集上随机取3张图片并进行预测。

-- FasterRCNN.py：算法文件


