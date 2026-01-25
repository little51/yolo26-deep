from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo26s-seg.pt") 
results = model("image01.png")
plt.imshow(results[0].plot())
plt.show()