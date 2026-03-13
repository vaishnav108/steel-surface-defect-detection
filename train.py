from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def main():

    # Check GPU
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("Using CPU")
        device = "cpu"

    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4
    )


if __name__ == "__main__":
    freeze_support()   # Required for Windows multiprocessing
    main()