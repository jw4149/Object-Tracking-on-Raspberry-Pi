import ultralytics
from picamera2 import Picamera2, Preview
from PIL import Image, ImageDraw
import cv2

ultralytics.checks()

from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

picam2 = Picamera2(camera_num=0)
picam2.start_preview(Preview.NULL)
config = picam2.create_preview_configuration({
    #"size": (320, 240),
    "size": (640, 480),
    "format": "BGR888"
})
picam2.configure(config)

picam2.start()

model = YOLO("yolov8n.pt")
# cap = cv2.VideoCapture("example.mp4")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
line_points = [(200, 20), (200, 460)]

# Video writer
'''
video_writer = cv2.VideoWriter("object_counting_output.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       20,
                       (320, 240))
'''


# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True
                )

while True:
    img = picam2.capture_array()
    
    tracks = model.track(img, persist=True, show=False)

    img = counter.start_counting(img, tracks)

    cv2.imwrite("output.png", img)
    # video_writer.write(img)
