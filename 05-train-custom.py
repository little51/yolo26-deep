from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo26s.pt")
results = model.train(data="mydataset/dataset.yaml",
                      device="cuda",
                      epochs=500,
                      imgsz=640,
                      batch=1,
                      workers=0)
results = model("image02.png")
plt.imshow(results[0].plot())
plt.show()
