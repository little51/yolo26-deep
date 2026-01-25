from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo26n-cls.pt")
results = model.train(
    data="mnist160", 
    epochs=100, 
    imgsz=64,
    batch=1,
    workers=0)
results = model.val()
results = model("image06.png")
plt.imshow(results[0].plot())
plt.show()