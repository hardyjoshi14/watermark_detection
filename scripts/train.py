from ultralytics import YOLO

def train_model(data_yaml, epochs=50, imgsz=640, batch=16):

    model = YOLO("yolov8n.pt")

    results = model.train(
        data = data_yaml,
        epochs = epochs,
        imgsz = imgsz,
        batch = batch,
        name = "watermark_detection"
    )
    model.export(format="onnx")

if __name__ ==  "__main__":

    data_yaml = "dataset.yaml"
    train_model(data_yaml, epochs=50, imgsz=640, batch=16)