import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import uuid

import csv
import os
import time

from ultralytics import YOLO
from easyocr import Reader

WEIGHTS = 'yolov8_lp_recognition/yolov8s.pt/weights/best.pt'
IMAGE = 'datasets/test/cars.jpg'
CONFIDENCE = 0.4
COLOUR = (255, 51, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONTSCALE = 0.75
THICKNESS = 2

def detect_plates(image, model):
    start = time.time()
    # run prediction
    detections = model.predict(image)[0].boxes.data

    # make sure predictions are not empty
    if detections.shape != torch.Size([0, 6]):
        
        boxes = []
        confidences = []

        # loop through detections
        for detection in detections:
            # pred example: [2.2895e+02, 7.6502e+02, 5.1979e+02, 8.5659e+02, 8.2725e-01, 0.0000e+00]
            # 0-3 = bbox coords, 4 = confidence, 5 = class
            confidence = detection[4]

            # skip if confidence is below threshold
            if float(confidence) < CONFIDENCE:
                continue

            # else record bbox coords + confidence
            boxes.append(detection[:4])
            confidences.append(detection[4])

        print(f"{len(boxes)} License plate(s) detected.")

        predictions = []
        images = []
        
        for i in range(len(boxes)):
            # extract bbox coords
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            # append to predictions
            predictions.append([[xmin, ymin, xmax, ymax], confidences[i]])
            print("INFO :: Detected regions", predictions)

            # draw predictions on original image
            bgr = cv.imread(IMAGE)
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            rgb_bbox = cv.rectangle(rgb, (xmin, ymin), (xmax, ymax), COLOUR, THICKNESS)
            text = "License: {:.2f}%".format(confidences[i] * 100)
            rgb_conf = cv.putText(rgb_bbox, text, (xmin, ymin - 5), FONT, FONTSCALE, COLOUR, THICKNESS)
            
            # crop plate region
            number_plate = rgb[ymin:ymax, xmin:xmax]

            images.append([rgb_conf, number_plate])


        end = time.time()
        print(f"Prediction took: {(end - start) * 1000:.0f} ms")
        
        return predictions, images


def ocr_plates(file_path, reader, predictions, write_to_csv=False):

    start = time.time()
    image = cv2.imread(file_path)

    for i, box in enumerate(predictions):
        # crop the number plate region
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]

        # detect the text from the license plate using the EasyOCR reader
        detection = reader.readtext(np_image, paragraph=True)

        if len(detection) == 0:
            # if no text is detected, set the `text` variable to an empty string
            text = ""
        else:
            # set the `text` variable to the detected text
            text = str(detection[0][1])

        # update the `predictions` list, adding the detected text
        number_plate_list[i].append(text)

    if write_to_csv:
        # open the CSV file
        csv_file = open("number_plates.csv", "w")
        # create a writer object
        csv_writer = csv.writer(csv_file)
        # write the header
        csv_writer.writerow(["image_path", "box", "text"])

        # loop over the `predictions` list
        for box, text in predictions:
            # write the image path, bounding box coordinates,
            # and detected text to the CSV file
            csv_writer.writerow([image_or_path, box, text])
        # close the CSV file
        csv_file.close()

    end = time.time()
    # show the time it took to recognize the number plates
    print(f"OCR took: {(end - start) * 1000:.0f} ms")

    return predictions
    


if __name__ == "__main__":

    # load best model weights
    model = YOLO(WEIGHTS)
    # initialize the EasyOCR reader
    reader = Reader(['en'], gpu=True)

    # path to an image or a video file
    file_path = IMAGE
    # get extensions to differentiate between images / videos
    _, file_extension = os.path.splitext(file_path)

    # Check extension
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print("INFO :: Image detection in progress")
        
        predictions, images = detect_plates(file_path, model)

        # if there are any number plates detected, recognize them
        if predictions != []:
            predictions = ocr_plates(file_path, reader, predictions, write_to_csv=True)

            for box, confidence in predictions:
                cv.putText(image, confidence, (box[0], box[3] + 15), FONT, FONTSCALE, COLOUR, THICKNESS, cv.LINE_AA)
            
            for image in images:
                # save images and region cutouts
                id = uuid.uuid4()
                cv.imwrite(f'datasets/pred/{id}_image.jpg', cv.cvtColor(image[0], cv.COLOR_BGR2RGB))
                cv.imwrite(f'datasets/pred/{id}_cutout.jpg', cv.cvtColor(image[1], cv.COLOR_BGR2RGB))
            

    # elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
    #     print("INFO :: Video detection in progress")

    #     video_cap = cv.VideoCapture(file_path)

    #     # grab the width and the height of the video stream
    #     frame_width = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    #     fps = int(video_cap.get(cv.CAP_PROP_FPS))
    #     # initialize the FourCC and a video writer object
    #     fourcc = cv.VideoWriter_fourcc(*"mp4v")
    #     writer = cv.VideoWriter("output.mp4", fourcc, fps,
    #                              (frame_width, frame_height))

    #     # loop over the frames
    #     while True:
    #         # starter time to computer the fps
    #         start = time.time()
    #         success, frame = video_cap.read()

    #         # if there is no more frame to show, break the loop
    #         if not success:
    #             print("There are no more frames to process."
    #                   " Exiting the script...")
    #             break

    #         predictions = detect_number_plates(frame, model)

    #         if predictions != []:
    #             predictions = recognize_number_plates(frame, reader,
    #                                                     predictions)

    #             for box, text in predictions:
    #                 cv.putText(frame, text, (box[0], box[3] + 15), FONT, FONTSCALE, COLOUR, THICKNESS, cv.LINE_AA)

    #         # end time to compute the fps
    #         end = time.time()
    #         # calculate the frame per second and draw it on the frame
    #         fps = f"FPS: {1 / (end - start):.2f}"
    #         cv.putText(frame, fps, (50, 50),
    #                     FONT, FONTSCALE, COLOUR, THICKNESS, cv.LINE_AA)

    #         # show the output frame
    #         cv.imshow("Output", frame)
    #         # write the frame to disk
    #         writer.write(frame)
    #         # if the 'q' key is pressed, break the loop
    #         if cv2.waitKey(1) == 27:  # Keep running until you press `esc`
    #           break

    #     # release the video capture, video writer, and close all windows
    #     video_cap.release()
    #     writer.release()
    #     cv.destroyAllWindows()
