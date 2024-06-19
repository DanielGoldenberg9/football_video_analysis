from ultralytics import YOLO

model = YOLO("yolov8x")
res = model.predict("Input_data/08fd33_4.mp4", save=True)

for box in res[0].boxes:
    print(box)
