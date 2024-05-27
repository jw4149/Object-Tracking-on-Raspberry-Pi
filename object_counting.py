
import argparse
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
import cv2
import tflite_runtime.interpreter as tflite

from picamera2 import Picamera2, Preview

from nms import non_max_suppression_yolov8

from sort import Sort
from tabulate import tabulate  # Add tabulate import

# How many coordinates are present for each box.
BOX_COORD_NUM = 4

def load_labels(filename):
  with open(filename, "r") as f:
    return [line.strip() for line in f.readlines()]


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
      "-c", 
      "--camera", type=int, help="Pi camera device to use")
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
      default="object_counting_output.mp4", 
      help="Video file to save model output"
  )

  args = parser.parse_args()

  if args.camera is not None:
    picam2 = Picamera2(camera_num=args.camera)
    picam2.start_preview(Preview.NULL)
    config = picam2.create_preview_configuration({
        "size": (320, 240),
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
  # id_pos = {}
  id_initial_pos = {}
  id_final_pos = {} 

  video_writer = cv2.VideoWriter(args.save_video,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      15,
                      (input_width, input_height))

  while True:
    if args.camera is None:
      if args.video is None:
        img = Image.open(args.image).resize((input_width, input_height))
      else:
        suc, img = cap.read()
        if not suc:
          break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize(
          size=(input_width, input_height), resample=Image.Resampling.LANCZOS)
    else:
      img = Image.fromarray(picam2.capture_array()).resize(
          size=(input_width, input_height), resample=Image.Resampling.LANCZOS)

    if args.save_input is not None:
      img.save("new_" + args.save_input)
      # Do a file move to reduce flicker in VS Code.
      shutil.move("new_" + args.save_input, args.save_input)

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
      input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]["index"], input_data)

    start_time = time.time()
    interpreter.invoke()
    # stop_time = time.time()

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

    # Clean up overlapping boxes. Reference:
    # https://petewarden.com/2022/02/21/non-max-suppressions-how-do-they-work/
    clean_boxes = non_max_suppression_yolov8(
        boxes, class_count, keypoint_count)

    if args.save_output is not None:
      img_draw = ImageDraw.Draw(img)
      # print(left_to_right)
      # print(right_to_left)
      y = 10
      img_draw.text((0, 0), "left to right:")
      for key, value in left_to_right.items():
        img_draw.text((0, y), f"{key}: {value}")
        y += 10
      y = 10
      img_draw.text((130, 0), "right to left:")
      for key, value in right_to_left.items():
        img_draw.text((130, y), f"{key}: {value}")
        y += 10

    for box in clean_boxes:
      center_x = box[0] * input_width
      center_y = box[1] * input_height
      w = box[2] * input_width
      h = box[3] * input_height
      half_w = w / 2
      half_h = h / 2
      box[0] = int(center_x - half_w)
      box[2] = int(center_x + half_w)
      box[1] = int(center_y - half_h)
      box[3] = int(center_y + half_h)
      box[4] = box[5]
    
    dets = np.array(clean_boxes).reshape(-1, 6)
    # print(dets)
    boxes = tracker.update(dets).tolist()

    for i in range(len(boxes)):
      box = boxes[i]
      left_x = box[0]
      bottom_y = box[1]
      right_x = box[2]
      top_y = box[3]
      center_x = (left_x + right_x)/2
      center_y = (bottom_y + top_y)/2
      class_index = int(box[4])
      class_label = class_labels[class_index]
      obj_id = box[5]

      # Store initial positions
      if obj_id not in id_initial_pos:
          id_initial_pos[obj_id] = (center_x, center_y)
      id_final_pos[obj_id] = (center_x, center_y)

      if args.save_output is not None:
          img_draw.rectangle(((left_x, top_y), (right_x, bottom_y)), fill=None)
          img_draw.text((center_x, center_y), f"{class_label} {obj_id}")

    # Remove IDs that are no longer detected
    tracked_ids = set(box[5] for box in boxes)
    for obj_id in list(id_initial_pos.keys()):
        if obj_id not in tracked_ids:
            initial_pos = id_initial_pos.pop(obj_id)
            final_pos = id_final_pos.pop(obj_id)
            initial_x, _ = initial_pos
            final_x, _ = final_pos
            if final_x > initial_x:
                left_to_right[class_label] = left_to_right.get(class_label, 0) + 1
            elif final_x < initial_x:
                right_to_left[class_label] = right_to_left.get(class_label, 0) + 1

    if args.save_output is not None:
      img.save("new_" + args.save_output)
      video_writer.write(cv2.UMat(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)))
      shutil.move("new_" + args.save_output, args.save_output)

    print("\n\n")
    # Print left_to_right and right_to_left in a table format
    table_data = [["Class", "Left to Right", "Right to Left"]]
    all_classes = set(left_to_right.keys()).union(set(right_to_left.keys()))
    for cls in all_classes:
        left_count = left_to_right.get(cls, 0)
        right_count = right_to_left.get(cls, 0)
        table_data.append([cls, left_count, right_count])

    stop_time = time.time()
    table_data.append(["Time: {:.3f} [ms]".format((stop_time - start_time) * 1000), "", ""])
    print(tabulate(table_data, headers="firstrow"))
    # print("time: {:.3f}ms".format((stop_time - start_time) * 1000))

    if args.camera is None and args.video is None:
      break
