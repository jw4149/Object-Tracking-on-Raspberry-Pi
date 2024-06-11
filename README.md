# Object Detection and Tracking with YOLOv8 and SORT on Raspberry Pi

This project involves object detection and tracking using YOLOv8 for detection and SORT for tracking. The implementation is for live camera feed for both baseline method and our method.

## Prerequisites

- Tested on Python 3.11.2
- Ensure you have the necessary dependencies installed by running:
  ```bash
  pip install -r requirements.txt
  ```
- Refer to reference section for understaning how the model is trained.

## Files

- `baseline.py`: Script for object detection and counting using YOLOv8 on video files.
- `baseline_cam.py`: Script for object detection and counting using YOLOv8 on live camera feed.
- `nms.py`: Non-max suppression implementation for YOLOv8 detections.
- `object_counting.py`: Script for object counting using YOLOv8 with TFLite model.
- `sort.py`: Implementation of the SORT tracking algorithm.
- `models/yolov8_labels.txt`: Labels for the YOLOv8 model.
- `models/yolov8n_int8.tflite`: Quantized YOLOv8 model in TFLite format.
- `doc/plot_frame_rate.py`: Testing script for analysing and comparing results.

## Usage
Ensure your camera is connected. Check the camera by running:
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
The default uses camera = 0

### Baseline Method For Live Camera Feed

Run the `baseline_cam.py` script:
   ```bash
   python baseline_cam.py
   ```

### Our Method For Live Camera Object Counting

 Run the `object_counting.py` script:
   ```bash
   python object_counting.py
   ```

## Reference
### GitHub Repositories
- [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort): This repository provides the implementation of the SORT algorithm used for object tracking in our project.
- [EE292D Lab 3](https://github.com/ee292d/labs/tree/main/lab3): This repository contains the trained model and some code we used in our project.

### Paper
For more details on the SORT algorithm, please refer to the following paper:
> **Simple Online and Realtime Tracking**  
> Alex Bewley, ZongYuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft  
> _CoRR, abs/1602.00763, 2016_  
> [arXiv link](http://arxiv.org/abs/1602.00763)

Here is the full citation in BibTeX format:
```bibtex
@article{DBLP:journals/corr/BewleyGORU16,
  author       = {Alex Bewley and
                  ZongYuan Ge and
                  Lionel Ott and
                  Fabio Ramos and
                  Ben Upcroft},
  title        = {Simple Online and Realtime Tracking},
  journal      = {CoRR},
  volume       = {abs/1602.00763},
  year         = {2016},
  url          = {http://arxiv.org/abs/1602.00763},
  eprinttype    = {arXiv},
  eprint       = {1602.00763},
  timestamp    = {Tue, 18 Oct 2022 08:35:41 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/BewleyGORU16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}