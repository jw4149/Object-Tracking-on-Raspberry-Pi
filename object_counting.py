import argparse
import shutil
import time
import signal
import sys
import pandas as pd # for report
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
import cv2
import tflite_runtime.interpreter as tflite

from picamera2 import Picamera2, Preview

from nms import non_max_suppression_yolov8

from sort import Sort
from tabulate import tabulate

# Suppress specific numpy warnings
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

# How many coordinates are present for each box.
BOX_COORD_NUM = 4

def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]

def draw_fading_trajectory(draw, trajectory, color, fade_length=20):
    fade_step = 255 // fade_length
    for i in range(1, len(trajectory)):
        start_point = trajectory[i - 1]
        end_point = trajectory[i]
        alpha = 255 - (len(trajectory) - i) * fade_step
        rgba_color = (*color, int(alpha))  # RGBA
        draw.line([start_point, end_point], fill=rgba_color, width=2)

def get_color_mapping(class_labels):
    np.random.seed(42)  # Ensure reproducibility
    return {label: tuple(np.random.randint(0, 256, 3)) for label in class_labels}

def signal_handler(args, start_time, frame_times, sig, frame):
    print('Interrupted! Exiting gracefully...')
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


    # Save log to Excel file
    if args.testing == "yes" or "Yes" or "YES" or "y" or "Y":
        data = {'Frame Time (s)': frame_times}
        df = pd.DataFrame(data)
        df['Frame Rate (fps)'] = 1 / df['Frame Time (s)']
        df.to_excel("frame_rate_log.xlsx", index=False)

    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        help="Input image")
    parser.add_argument(
        "-v",
        "--video",
        help="Input video")
    parser.add_argument(
        "-m",
        "--model_file",
        default="models/yolov8n_int8.tflite",
        help="TF Lite model to be executed")
    parser.add_argument(
        "-l",
        "--label_file",
        default="models/yolov8_labels.txt",
        help="name of file containing labels")
    parser.add_argument(
        "--input_mean",
        default=0.0, type=float,
        help="input_mean")
    parser.add_argument(
        "--input_std",
        default=255.0, type=float,
        help="input standard deviation")
    parser.add_argument(
        "--num_threads", default=2, type=int, help="number of threads")
    parser.add_argument(
        "--camera", default=0, type=int, help="Pi camera device to use")
    parser.add_argument(
        "--save_input", default=None, help="Image file to save model input to")
    parser.add_argument(
        "--save_output", default="output.png", help="Image file to save model output to")
    parser.add_argument(
        "--score_threshold",
        default=0.6, type=float,
        help="Score level needed to include results")
    parser.add_argument(
        "--output_format",
        default="yolov8_detect",
        help="How to interpret the output from the model")
    parser.add_argument(
        "--save_video", 
        default=None, 
        help="Video file to save model output")
    parser.add_argument(
        "--testing", 
        default=0,
        type=int,
        help="Testing time in seconds for analysis")
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        help="Resize the input image to a square image of this size"
    )

    args = parser.parse_args()
    start_time = time.time()
    frame_times = []

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(args, start_time, frame_times, sig, frame))


    if args.camera is not None:
        picam2 = Picamera2(camera_num=args.camera)
        picam2.start_preview(Preview.NULL)
        config = picam2.create_preview_configuration({
            "size": (640, 480),  # Double the size
            "format": "BGR888"
        })
        picam2.configure(config)
        picam2.start()
    elif args.video is not None:
        cap = cv2.VideoCapture(args.video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    tracker = Sort()

    interpreter = tflite.Interpreter(
        model_path=args.model_file,
        num_threads=args.num_threads)
    interpreter.allocate_tensors()

    class_labels = load_labels(args.label_file)
    color_mapping = get_color_mapping(class_labels)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]["dtype"] == np.float32

    # NxHxWxC, H:1, W:2
    input_height = input_details[0]["shape"][1]
    input_width = input_details[0]["shape"][2]

    max_box_count = output_details[0]["shape"][2]

    if args.output_format == "yolov8_detect":
        class_count = output_details[0]["shape"][1] - BOX_COORD_NUM
        keypoint_count = 0
    else:
        print(f"Unknown output format {args.output_format}")
        exit(0)

    if len(class_labels) != class_count:
        print("Model has %d classes, but %d labels" %
              (class_count, len(class_labels)))
        exit(0)

    left_to_right = {}
    right_to_left = {}
    trajectory_history = {}
    counted_objects = set()


    input_width_show = input_width
    input_height_show = input_height
    if args.resize is not None:
        input_width_show = input_width * args.resize
        input_height_show = input_width * args.resize
        
    if args.save_video is not None:
        video_name = "object_counting_output_" + args.save_video + ".mp4"

        video_writer = cv2.VideoWriter(video_name,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    15,
                                    (input_width_show, input_height_show))  # Double the size

    vertical_line_x = (input_width_show) // 2  # Double the size

    while True:
        frame_start_time = time.time()

        if args.camera is None:
            if args.video is None:
                img = Image.open(args.image).resize((input_width_show, input_height_show))
            else:
                suc, img = cap.read()
                if not suc:
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img).resize(
                    size=(input_width_show, input_height_show), resample=Image.Resampling.LANCZOS)
        else:
            img = Image.fromarray(picam2.capture_array()).resize(
                size=(input_width_show, input_height_show), resample=Image.Resampling.LANCZOS)

        # Resize to model input size
        img_resized = img.resize((input_width, input_height))

        if args.save_input is not None:
            img_resized.save("new_" + args.save_input)
            shutil.move("new_" + args.save_input, args.save_input)

        # add N dim
        input_data = np.expand_dims(img_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - args.input_mean) / args.input_std

        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        results = np.squeeze(output_data).transpose()

        boxes = []
        for i in range(max_box_count):
            raw_box = results[i]
            center_x = raw_box[0]
            center_y = raw_box[1]
            w = raw_box[2]
            h = raw_box[3]
            if args.output_format == "yolov8_detect":
                class_scores = raw_box[BOX_COORD_NUM:]
                for index, score in enumerate(class_scores):
                    if (score > args.score_threshold):
                        boxes.append([center_x, center_y, w, h, score, index])
            else:
                print(f"Unknown output format {args.output_format}")

        print("\n\n")
        table_data = [["Class", "Left to Right", "Right to Left"]]
        all_classes = set([v for k, v in left_to_right.items()]).union(set([v for k, v in right_to_left.items()]))
        
        y_l = y_r = 10
        if args.save_output is not None:
            # Print object counts on the image
            img_draw = ImageDraw.Draw(img)
            img_draw.text((0, 0), "left to right:")
            img_draw.text((130, 0), "right to left:")

        for cls in all_classes:
            left_count = sum(1 for v in left_to_right.values() if v == cls)
            right_count = sum(1 for v in right_to_left.values() if v == cls)
            table_data.append([cls, left_count, right_count])

            if args.save_output is not None:
                # print(left_to_right)
                if left_count > 0:
                    # right_count = sum(1 for v in right_to_left.values() if v == cls)
                    img_draw.text((0, y_l), f"{cls}: {left_count}")
                    y_l += 10
                # print(right_to_left)  
                if right_count > 0:
                    # right_count = sum(1 for v in right_to_left.values() if v == cls)
                    img_draw.text((130, y_r), f"{cls}: {right_count}")
                    y_r += 10

        clean_boxes = non_max_suppression_yolov8(
            boxes, class_count, keypoint_count)
            
        for box in clean_boxes:
            center_x = box[0] * (input_width_show)
            center_y = box[1] * (input_height_show)
            w = box[2] * (input_width_show)
            h = box[3] * (input_height_show)
            half_w = w / 2
            half_h = h / 2
            box[0] = int(center_x - half_w)
            box[2] = int(center_x + half_w)
            box[1] = int(center_y - half_h)
            box[3] = int(center_y + half_h)
            box[4] = box[5]

        dets = np.array(clean_boxes).reshape(-1, 6)
        boxes = tracker.update(dets).tolist()

        img_draw = ImageDraw.Draw(img)

        # Draw the virtual vertical line
        img_draw.line([(vertical_line_x, 0), (vertical_line_x, input_height_show)], fill="blue", width=2)

        for i in range(len(boxes)):
            box = boxes[i]
            left_x = box[0]
            bottom_y = box[1]
            right_x = box[2]
            top_y = box[3]
            center_x = (left_x + right_x) / 2
            center_y = (bottom_y + top_y) / 2
            class_index = int(box[4])
            class_label = class_labels[class_index]
            obj_id = box[5]

            color = color_mapping[class_label]

            if obj_id not in trajectory_history:
                trajectory_history[obj_id] = deque(maxlen=20)
            trajectory_history[obj_id].append((center_x, center_y))

            # Draw the trajectory with fading effect
            draw_fading_trajectory(img_draw, list(trajectory_history[obj_id]), color)

            if args.save_output is not None:
                img_draw.rectangle(((left_x, bottom_y), (right_x, top_y)), outline=color, width=2)
                img_draw.text((center_x, center_y), f"{class_label} {obj_id}", fill=color)

            # Check if the object has crossed the line for the first time within the most recent (max 20) frames
            if len(trajectory_history[obj_id]) > 1 and obj_id not in counted_objects:
                if trajectory_history[obj_id][0][0] < vertical_line_x and trajectory_history[obj_id][-1][0] > vertical_line_x:
                    left_to_right[obj_id] = class_label
                    counted_objects.add(obj_id)
                elif trajectory_history[obj_id][0][0] > vertical_line_x and trajectory_history[obj_id][-1][0] < vertical_line_x:
                    right_to_left[obj_id] = class_label
                    counted_objects.add(obj_id)

        if args.save_output is not None:
            img.save("new_" + args.save_output)
            if args.save_video is not None:
                video_writer.write(cv2.UMat(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)))
                video_writer.write(cv2.UMat(np.array(img)))
            shutil.move("new_" + args.save_output, args.save_output)

        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time

        if args.testing != 0:
            runtime = frame_end_time - start_time
            frame_times.append(frame_time)
            print(runtime, len(frame_times))
            # if  runtime >= args.testing:
            if  len(frame_times) >= args.testing:
                break

        table_data.append(["Time: {:.3f} [ms]".format(frame_time * 1000), "", ""])
        print(tabulate(table_data, headers="firstrow"))

        if args.camera is None and args.video is None:
            break

    if args.testing != 0:
        # Calculate frame rate
        end_time = time.time()
        elapsed_time = end_time - start_time
        frame_rate = len(frame_times) / elapsed_time
        print(f"Average frame rate: {frame_rate:.2f} frames per second")

        # Save log to Excel file
        data = {'Frame Time (s)': frame_times}
        df = pd.DataFrame(data)
        df['Frame Rate (fps)'] = 1 / df['Frame Time (s)']
        df.to_excel("doc/frame_rate_log_ours.xlsx", index=False)