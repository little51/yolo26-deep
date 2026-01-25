from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo26s-seg.pt")

results = model.train(
    data="carparts-seg.yaml", 
    epochs=100, 
    imgsz=640,
    batch=1,
    workers=0)
results = model.val()
results = model("image04.png")
plt.imshow(results[0].plot())
plt.show()