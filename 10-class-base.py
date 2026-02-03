from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
results = model("image07.png")

result = results[0]
probs = result.probs

confidence_threshold = 0.4  # 只显示概率大于40%的类别

print(f"显示概率 > {confidence_threshold:.0%} 的类别:")
for idx, name in result.names.items():
    prob = float(probs.data[idx])
    if prob > confidence_threshold:
        print(f"  {name}: {prob:.2%}")