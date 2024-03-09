import PIL.Image as Image
import gradio as gr

from ultralytics import ASSETS, YOLO

WEIGHTS = "yolov8_lp_recognition/yolov8s.pt/weights/best.pt"

model = YOLO(WEIGHTS)


def find_license_plate(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=find_license_plate,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="License Plate Detector",
    description=WEIGHTS,
    examples=[
        [ASSETS / "/opt/app/notebooks/datasets/test/lamborghini-huracan-sterrato_1.jpg", 0.25, 0.45],
        [ASSETS / "/opt/app/notebooks/datasets/test/lamborghini-huracan-sterrato_2.jpg", 0.25, 0.45],
        [ASSETS / "/opt/app/notebooks/datasets/test/cars.jpg", 0.25, 0.45]
    ]
)

if __name__ == '__main__':
    iface.launch(server_name="0.0.0.0", server_port=7861) 
