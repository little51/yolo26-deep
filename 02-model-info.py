from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")

# 查看模型信息
print("模型类别:", model.names)  # 输出类别名称字典
print("总类别数:", len(model.names))  # 输出类别数量
print("模型信息:")
print(model.model)  # 查看模型结构