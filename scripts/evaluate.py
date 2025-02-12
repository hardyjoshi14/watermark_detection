from ultralytics import YOLO

def evaluate_yolo(model_path, data_yaml):
    
    model = YOLO(model_path)

    # Evaluate the model on the validation set
    results = model.val(data=data_yaml)

    # Print evaluation metrics
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")
    print(f"mAP50: {results.box.map50}")
    print(f"F1 Score: {results.box.f1}")
    
if __name__ == "__main__":
    # Path to the trained model
    model_path = "runs/detect/watermark_detection/weights/best.pt"

    # Path to the dataset YAML file
    data_yaml = "dataset.yaml"

    # Evaluate the model
    evaluate_yolo(model_path, data_yaml)