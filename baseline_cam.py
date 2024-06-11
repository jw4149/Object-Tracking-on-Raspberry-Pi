import argparse
import signal
import sys
import pandas as pd
import time

import ultralytics
from picamera2 import Picamera2, Preview
from PIL import Image, ImageDraw
import cv2

ultralytics.checks()

from ultralytics import YOLO
from ultralytics.solutions import object_counter

def signal_handler(start_time, frame_times, sig, frame, testing):
    print('Interrupted! Exiting gracefully...')

    if testing:
        # Save log to Excel file
        data = {'Frame Time (s)': frame_times}
        df = pd.DataFrame(data)
        df['Frame Rate (fps)'] = 1 / df['Frame Time (s)']
        df.to_excel("frame_rate_log.xlsx", index=False)

    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing", default=0, type=int, help="Enable frame rate logging and export to Excel")
    args = parser.parse_args()

    start_time = time.time()
    frame_times = []

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(start_time, frame_times, sig, frame, args.testing))

    picam2 = Picamera2(camera_num=0)
    picam2.start_preview(Preview.NULL)
    config = picam2.create_preview_configuration({
        "size": (320, 240),
        # "size": (640, 480),
        "format": "RGB888"
    })
    picam2.configure(config)
    picam2.start()

    model = YOLO("yolov8n.pt")

    # Define line points
    line_points = [(200, 20), (200, 460)]

    # Init Object Counter
    counter = object_counter.ObjectCounter(view_img=True,
                     reg_pts=line_points,
                     classes_names=model.names,
                     draw_tracks=True)

    while True:
        frame_start_time = time.time()
        
        img = picam2.capture_array()
        
        tracks = model.track(img, persist=True, show=False)
        img = counter.start_counting(img, tracks)
        
        cv2.imwrite("output.png", img)
        
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        
        if args.testing != 0:
            frame_times.append(frame_time)
            if len(frame_times) >= args.testing:
                break

    if args.testing != 0:
        # After loop, calculate frame rate and save to Excel
        end_time = time.time()
        elapsed_time = end_time - start_time
        frame_rate = len(frame_times) / elapsed_time
        print(f"Average frame rate: {frame_rate:.2f} frames per second")

        # Save log to Excel file
        data = {'Frame Time (s)': frame_times}
        df = pd.DataFrame(data)
        df['Frame Rate (fps)'] = 1 / df['Frame Time (s)']
        df.to_excel("doc/frame_rate_log_baseline.xlsx", index=False)
