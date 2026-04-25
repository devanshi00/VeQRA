# Downloaded the Dataset from Kaggle --> https://www.kaggle.com/datasets/redzapdos123/dior-r-dataset-yolov11-obb-format
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo11x-obb.pt")

model.train(
    data="YOLODIOR-R/data.yaml",
    epochs=50,
    batch=32, # For 40GB A100 GPU
    device=0,
    name="dior_experiment_1",
    imgsz=512,
    project="root/dior_runs",
    lr0=0.005,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    optimizer="AdamW",
    workers=8,
    augment=True,
    rect=True,
    mosaic=True,
    mixup=True,
    label_smoothing=0.1,
    val=True,
    save_period=1,
)
