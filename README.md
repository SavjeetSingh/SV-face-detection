# Real-Time Face Detection

This repository provides a Python script for real-time face detection using OpenCV's DNN module and a pre-trained Caffe model.

## Features

- Detects faces from a webcam stream in real-time.
- Uses the Single Shot Multibox Detector (SSD) with ResNet-10 architecture.
- Draws bounding boxes and confidence scores on detected faces.

## Files

- [`real-time-detect.py`](real-time-detect.py): Main script for real-time face detection.
- [`deploy.prototxt`](deploy.prototxt): Model architecture definition.
- [`res10_300x300_ssd_iter_140000.caffemodel`](res10_300x300_ssd_iter_140000.caffemodel): Pre-trained weights.
- [`gojo-face.jpg`](gojo-face.jpg): Example image (optional).
- [`detect.py`](detect.py): Additional detection script (optional).

## Requirements

- Python 3.13+
- OpenCV (`cv2`)
- NumPy

You can use the provided virtual environment in the [`cv-venv`](cv-venv) folder.

## Setup

1. **Install dependencies** (if not using the provided virtual environment):

    ```sh
    pip install opencv-python numpy
    ```

2. **Run the script**:

    ```sh
    python real-time-detect.py
    ```

3. **Quit**: Press `q` in the video window to exit.

## Model Details

- The model is defined in [`deploy.prototxt`](deploy.prototxt) and uses SSD for face detection.
- Pre-trained weights are loaded from [`res10_300x300_ssd_iter_140000.caffemodel`](res10_300x300_ssd_iter_140000.caffemodel).

## References

- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [Caffe Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

