# Steel Surface Defect Detection using YOLOv8

This project detects defects on steel surfaces using a deep learning object detection model (YOLOv8).

## Dataset
NEU Surface Defect Database

## Defect Classes
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Model
YOLOv8 (Ultralytics)

## Project Structure

steel-surface-defect-detection
│
├── train.py          # Model training
├── evaluate.py       # Model evaluation
├── predict.py        # Defect prediction
├── data.yaml         # Dataset configuration
├── yolo_dataset      # Dataset
├── test_images       # Images used for testing
└── predictions       # Output images

## How to Run

### Train the Model
