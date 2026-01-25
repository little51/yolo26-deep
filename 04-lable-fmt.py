import json
import requests
from pathlib import Path
import random
import time
from urllib.parse import urlparse

def convert_ndjson_to_yolo(ndjson_content, output_dir="./mydataset", train_ratio=0.7):
    """将NDJSON格式数据集转换为YOLO格式，按比例分割训练集和验证集"""
    
    output_dir = Path(output_dir)
    
    # 创建目录结构
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    class_names = {}
    image_data = []
    
    # 第一遍：收集所有图像信息
    for line in ndjson_content.strip().split('\n'):
        if not line:
            continue
            
        data = json.loads(line)
        
        if data.get("type") == "dataset":
            class_names = data.get("class_names", {})
        elif data.get("type") == "image":
            image_data.append(data)
    
    # 随机打乱图像顺序
    random.shuffle(image_data)
    
    # 按比例分割数据集
    split_index = int(len(image_data) * train_ratio)
    train_images = image_data[:split_index]
    val_images = image_data[split_index:]
    
    # 处理训练集
    train_counter = 1
    for data in train_images:
        new_filename = _process_image(data, output_dir, "train", train_counter)
        if new_filename:
            train_counter += 1
    
    # 处理验证集
    val_counter = 1
    for data in val_images:
        new_filename = _process_image(data, output_dir, "val", val_counter)
        if new_filename:
            val_counter += 1
    
    # 创建YAML配置文件
    if class_names:
        sorted_classes = sorted([(int(k), v) for k, v in class_names.items()])
        class_list = [name for _, name in sorted_classes]
        
        yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {len(class_list)}
names: {class_list}
"""
        
        (output_dir / "dataset.yaml").write_text(yaml_content)
    
    print(f"转换完成！")
    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(val_images)} 张图像")
    print(f"数据集保存在: {output_dir}")
    return output_dir

def _process_image(data, output_dir, split_type, counter):
    """处理单个图像并保存到指定文件夹"""
    file_name = data.get("file", "")
    image_url = data.get("url", "")
    annotations = data.get("annotations", {}).get("boxes", [])
    
    if not file_name or not image_url:
        return None
    
    # 生成新文件名（根据你的需求选择一种方案）
    
    # 方案1: 使用序号 + 时间戳
    timestamp = int(time.time() * 1000) % 1000000
    new_filename = f"{split_type}_{counter:04d}_{timestamp}.jpg"
    
    # 下载图像到新文件名
    img_path = output_dir / "images" / split_type / new_filename
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        img_path.write_bytes(response.content)
    except Exception as e:
        print(f"下载失败 {file_name}: {e}")
        return None
    
    # 保存标注文件，使用相同的文件名（不含扩展名）
    label_filename = f"{Path(new_filename).stem}.txt"
    label_path = output_dir / "labels" / split_type / label_filename
    
    with open(label_path, 'w') as f:
        for ann in annotations:
            if len(ann) >= 5:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    print(f"✓ 保存: {new_filename}")
    return new_filename

if __name__ == "__main__":
    # 从文件读取内容
    with open('office.ndjson', 'r', encoding='utf-8') as f:
        ndjson_content = f.read()
    
    # 可以指定输出目录
    dataset_dir = convert_ndjson_to_yolo(
        ndjson_content,
        output_dir="./mydataset",
        train_ratio=0.8
    )