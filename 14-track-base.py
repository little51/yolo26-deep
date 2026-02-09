from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.track("test.mp4", show=True)
