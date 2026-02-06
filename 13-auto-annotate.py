from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="custom_dataset/train/dress",
    det_model="yolo26n.pt",
    sam_model="mobile_sam.pt",
    device="cuda",
    output_dir="custom_dataset/labels/train/dress"
)

# visualize results
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = YOLO("yolo26n.pt")
label_map = model.model.names

def visualize_polygon_annotations(image_path, polygon_label_path, label_map):
    """
    可视化多边形分割标注
    """
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 读取多边形标签
    with open(polygon_label_path, 'r') as f:
        lines = f.readlines()
    
    # 为不同类别生成不同颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
    ]
    
    for i, line in enumerate(lines):
        parts = list(map(float, line.strip().split()))
        if len(parts) < 3:
            continue
            
        class_id = int(parts[0])
        polygon_points = parts[1:]
        
        # 将归一化坐标转换为像素坐标
        points = []
        for j in range(0, len(polygon_points), 2):
            if j + 1 < len(polygon_points):
                x = polygon_points[j] * w
                y = polygon_points[j + 1] * h
                points.append([x, y])
        
        if len(points) < 3:  # 至少需要3个点才能形成多边形
            continue
        
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # 绘制多边形
        color = colors[class_id % len(colors)]
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
        
        # 填充多边形（半透明）
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(mask, [points], color)
        img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        
        # 添加类别标签
        if len(points) > 0:
            class_name = label_map.get(class_id, f"Class {class_id}")
            # 找到多边形的中心
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img, class_name, (cx, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 显示图像
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Polygon Segmentation Annotations")
    plt.show()

# 可视化多边形
visualize_polygon_annotations(
    "custom_dataset/train/dress/3eeaf330-2460-4d7f-844f-d7254d12e587.jpg", 
    "custom_dataset/labels/train/dress/3eeaf330-2460-4d7f-844f-d7254d12e587.txt", 
    label_map,
)