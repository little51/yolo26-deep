# YOLO26深度讲解

## 一、实验环境

```shell
# 创建虚拟环境
conda create -n yolo26 python=3.12 -y
# 激活虚拟环境
conda activate yolo26
# 安装最新YOLO库
pip install ultralytics -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装PyTorch（仅Windows）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## 二、视觉任务

```shell
# 1、目标检测基础
python 01-detect-base.py
# 2、模型参数
python 02-model-info.py
# 3、目标检测训练
python 03-train-base.py
# 4、训练数据标注
python 04-lable-fmt.py
# 5、自定义数据训练
python 05-train-custom.py
# 6、自定义类别检测
python 06-detect-custom.py
# 7、语义分割基础
python 07-segment-base.py
# 8、语义分割训练
python 08-segment-train.py
# 9、开放词汇分割
python 09-segment-yoloe.py
# 10、分类基础
python 10-class-base.py
# 11、分类训练
python 11-class-train.py
```

![](https://gitclone.com/download1/aliendao/weixin-aliendao.jpg)
