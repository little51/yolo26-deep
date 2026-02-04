from ultralytics import YOLO

model = YOLO('yolo26n-cls.pt') 

results = model.train(
    data='custom_dataset',
    epochs=100,
    imgsz=224,
    batch=32, 
    workers=0
)

val_results = model.val()

# 输出验证结果
print(f"验证准确率: {val_results.top1:.2%}")
print(f"Top-5准确率: {val_results.top5:.2%}")
print(f"损失值: {val_results.loss:.4f}")