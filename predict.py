from ultralytics import YOLO
import cv2
import os
import numpy as np
from multiprocessing import freeze_support

MODEL_PATH = "runs/detect/train3/weights/best.pt"
INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "predictions"
CONF_THRESHOLD = 0.40


def is_metal_surface(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return saturation < 60


def main():

    model = YOLO(MODEL_PATH)

    if not os.path.exists(INPUT_FOLDER):
        print("Input folder not found:", INPUT_FOLDER)
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for file in os.listdir(INPUT_FOLDER):

        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        image_path = os.path.join(INPUT_FOLDER, file)

        img = cv2.imread(image_path)

        if img is None:
            print("Skipping unreadable file:", file)
            continue

        print("\nProcessing:", file)

        if not is_metal_surface(img):
            print("⚠ Not likely a metal surface")

        results = model(img)

        boxes = results[0].boxes
        valid_detection = False

        if boxes is not None:
            for box in boxes:
                if float(box.conf) > CONF_THRESHOLD:
                    valid_detection = True

        if valid_detection:
            print("⚠ Defect detected")
            result_img = results[0].plot()
        else:
            print("No defect detected")
            result_img = img

        output_path = os.path.join(OUTPUT_FOLDER, file)
        cv2.imwrite(output_path, result_img)

        print("Saved:", output_path)


if __name__ == "__main__":
    freeze_support()
    main()
    