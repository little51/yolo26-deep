from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo26n-cls.pt")
results = model.train(
    data="mnist160", 
    epochs=1000, 
    imgsz=64,
    batch=1,
    workers=0)
results = model.val()
results = model("image06.png")
print(f"预测类别: {results[0].probs.top1}")
print(f"置信度: {results[0].probs.top1conf:.2f}")
print(f"所有类别概率: {results[0].probs.data.tolist()}")