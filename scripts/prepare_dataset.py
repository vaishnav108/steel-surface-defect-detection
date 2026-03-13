import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "dataset"

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

# YOLO output folders
for folder in [
    "yolo_dataset/images/train",
    "yolo_dataset/images/val",
    "yolo_dataset/labels/train",
    "yolo_dataset/labels/val"
]:
    os.makedirs(folder, exist_ok=True)

for class_id, class_name in enumerate(classes):

    class_path = os.path.join(dataset_path, class_name)

    images = [
        f for f in os.listdir(class_path)
        if f.endswith((".jpg",".png",".jpeg",".bmp"))
    ]

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img in train_imgs:

        src = os.path.join(class_path, img)
        dst = os.path.join("yolo_dataset/images/train", img)

        shutil.copy(src, dst)

        label_file = os.path.join("yolo_dataset/labels/train", img.split(".")[0] + ".txt")

        with open(label_file, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0")

    for img in val_imgs:

        src = os.path.join(class_path, img)
        dst = os.path.join("yolo_dataset/images/val", img)

        shutil.copy(src, dst)

        label_file = os.path.join("yolo_dataset/labels/val", img.split(".")[0] + ".txt")

        with open(label_file, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0")

print("Dataset successfully converted to YOLO format.")