import os
import cv2

image_folder = "yolo_dataset/images/train"
label_folder = "yolo_dataset/labels/train"

for image_file in os.listdir(image_folder):

    if not image_file.endswith((".jpg", ".png", ".jpeg", ".bmp")):
        continue

    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, image_file.split(".")[0] + ".txt")

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    if os.path.exists(label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            class_id, x_center, y_center, width, height = map(float, line.split())

            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Label Visualization", img)

    key = cv2.waitKey(0)

    if key == 27:  # press ESC to exit
        break

cv2.destroyAllWindows()