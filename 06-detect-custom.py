from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("runs/detect/train/weights/best.pt")
results = model("image02.png")
plt.imshow(results[0].plot())
plt.show()