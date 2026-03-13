from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("runs/detect/train3/weights/best.pt")

    metrics = model.val(
        data="data.yaml",
        imgsz=640,
        batch=16,
        device=0,
        workers=2
    )

    print(metrics)


if __name__ == "__main__":
    freeze_support()
    main()