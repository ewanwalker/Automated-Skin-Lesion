from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset/",
    epochs=50,
    imgsz=224,
    batch=16,
    patience=10,       
    augment=True,      
)

metrics = model.val()
print(metrics.top1)   # top-1 accuracy