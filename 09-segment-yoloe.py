from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yoloe-26l-seg.pt") 
names = ["person", "black car","computer"]
model.set_classes(names, model.get_text_pe(names))
results = model.predict("image05.png")
plt.imshow(results[0].plot())
plt.show()