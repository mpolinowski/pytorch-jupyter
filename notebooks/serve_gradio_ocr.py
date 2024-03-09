import time
import cv2 as cv
import os
import csv

import PIL.Image as Image
import matplotlib.image as mpimg
import gradio as gr

from ultralytics import ASSETS, YOLO
import torch
# from easyocr import Reader

WEIGHTS = 'yolov8_lp_recognition/yolov8s.pt/weights/best.pt'
model = YOLO(WEIGHTS)

COLOUR = (255, 51, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
THICKNESS = 2

# bounding box coords collected we now can use them to cut out the license plate
def draw_bounding_boxes(img, conf_threshold, iou_threshold):
    start = time.time()

    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    detections = results[0].boxes.data
    
    # check if nothing was detected
    if detections.shape != torch.Size([0, 6]):
        boxes = []
        confidences = []
        classes = []
        # if detection loop through and get bboxes
        for detection in detections:

            bbox = detection[:4]
            confidence = detection[4]
            type = int(detection[5])

            # ignore if below confidence threshold
            if float(confidence) < conf_threshold:
                continue

            # collect params
            boxes.append(bbox)
            confidences.append(confidence)
            classes.append(type)

        # draw bbox over detected plate
        plate_boxes = []
        detections = []

        for i in range(len(boxes)):
            # separate x/y bbox coords
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            plate_boxes.append([[xmin, ymin, xmax, ymax]])
            
            img_bgr = cv.imread(img)
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

            image = cv.rectangle(img_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), COLOUR, THICKNESS)
            text = "Score: {:.2f}%".format(confidences[i] * 100)
            image = cv.putText(image, text, (int(xmin), int(ymin) - 5), FONT,  
                   FONTSCALE, COLOUR, THICKNESS, cv.LINE_AA)
            
            # cut out plate
            plate = img_rgb[ymin:ymax, xmin:xmax]
            detections.append(image)
            detections.append(plate)

        end = time.time()
        print("Detected plates: ", plate_boxes)
        print(f"Detection took: {(end - start)*1000:.0f}ms")

    return detections[0]


iface = gr.Interface(
    fn=draw_bounding_boxes,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="numpy", label="Result"),
    title="License Plate Detector",
    description=WEIGHTS,
    examples=[
        [ASSETS / "/run/media/xiaodie/dev/pytorch-jupyter/notebooks/datasets/test/lamborghini-huracan-sterrato_1.jpg", 0.25, 0.45],
        [ASSETS / "/run/media/xiaodie/dev/pytorch-jupyter/notebooks/datasets/test/lamborghini-huracan-sterrato_2.jpg", 0.25, 0.45],
        [ASSETS / "/run/media/xiaodie/dev/pytorch-jupyter/notebooks/datasets/test/cars.jpg", 0.25, 0.45]
    ]
    # examples=[
    #     [ASSETS / "/opt/app/notebooks/datasets/test/lamborghini-huracan-sterrato_1.jpg", 0.25, 0.45],
    #     [ASSETS / "/opt/app/notebooks/datasets/test/lamborghini-huracan-sterrato_2.jpg", 0.25, 0.45],
    #     [ASSETS / "/opt/app/notebooks/datasets/test/cars.jpg", 0.25, 0.45]
    # ]
)

if __name__ == '__main__':
    iface.launch() 

# if __name__ == '__main__':
#     iface.launch(server_name="0.0.0.0", server_port=7861) 