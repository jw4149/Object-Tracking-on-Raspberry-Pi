# Object Detection and Tracking with YOLOv8 and SORT

This project involves object detection and tracking using YOLOv8 for detection and SORT for tracking. The implementation is designed for both video files and live camera feeds for baseline method, and so far live camera feed only for SORT method.

## Prerequisites

- Python 3.8 or later
- Ensure you have the necessary dependencies installed by running:
  ```bash
  pip install -r requirements.txt
  ```

## Files

- `baseline.py`: Script for object detection and counting using YOLOv8 on video files.
- `baseline_cam.py`: Script for object detection and counting using YOLOv8 on live camera feed.
- `nms.py`: Non-max suppression implementation for YOLOv8 detections.
- `object_counting.py`: Script for object counting using YOLOv8 with TFLite model.
- `sort.py`: Implementation of the SORT tracking algorithm.
- `models/yolov8_labels.txt`: Labels for the YOLOv8 model.
- `models/yolov8n_int8.tflite`: Quantized YOLOv8 model in TFLite format.

## Usage

### Baseline Method For Video Files

1. Place your video file in the working directory.
2. Run the `baseline.py` script:
   ```bash
   python baseline.py
   ```

### Baseline Method For Live Camera Feed

1. Ensure your camera is connected.
2. Run the `baseline_cam.py` script:
   ```bash
   python baseline_cam.py
   ```

### Our Method For Live Camera Object Counting

1. Ensure your camera is connected. Check the camera by running:
```bash
rpicam-hello --list-cameras
```

You should see results like this:

```bash
Available cameras
-----------------
0 : imx219 [3280x2464 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx219@10)
    Modes: 'SRGGB10_CSI2P' : 640x480 [206.65 fps - (1000, 752)/1280x960 crop]
                             1640x1232 [41.85 fps - (0, 0)/3280x2464 crop]
                             1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                             3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
           'SRGGB8' : 640x480 [206.65 fps - (1000, 752)/1280x960 crop]
                      1640x1232 [83.70 fps - (0, 0)/3280x2464 crop]
                      1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                      3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
```
2. Run the script with the desired parameters:
   ```bash
   python object_counting.py --camera=0
   ```
