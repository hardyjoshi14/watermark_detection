from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import base64


app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO("runs/detect/watermark_detection/weights/best.pt")

def generate_heatmap(image, boxes):
    """
    Generate a heatmap overlay for the detected watermarks.
    """
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float32)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        heatmap[y1:y2, x1:x2] += 1  # Increment heatmap intensity in the bounding box area

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to [0, 255]
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)  # Apply color map
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)  # Overlay heatmap on the image
    
    return overlay


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image is provided in the request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Read the image file
        file = request.files["image"]
        file_bytes = file.read()
        image = Image.open(io.BytesIO(file_bytes))

        # Determine input image format
        image_format = file.filename.split(".")[-1].lower()  # Get file extension
        valid_formats = ["jpg", "jpeg", "png"]

        if image_format not in valid_formats:
            return jsonify({"error": f"Unsupported file format: {image_format}"}), 400

        # Convert image to numpy array
        image = np.array(image)

        # Handle PNG images with alpha channel (RGBA)
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Perform inference
        results = model(image)

        # Parse the results
        watermarks = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
            for box in boxes:
                x1, y1, x2, y2 = box
                watermarks.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(result.boxes.conf.cpu().numpy()[0])
                })

        # Generate heatmap overlay 
        heatmap_overlay = None
        if watermarks:
            heatmap_overlay = generate_heatmap(image, [box["bbox"] for box in watermarks])
            print(f"Encoding heatmap overlay as {image_format}")
            print(f"Image shape: {heatmap_overlay.shape}, dtype: {heatmap_overlay.dtype}")
            try:
                _, buffer = cv2.imencode(f".{image_format}" , heatmap_overlay)  # Encode correctly
                heatmap_overlay = base64.b64encode(buffer).decode("utf-8")  # Convert to base64
            except Exception as e:
                print(f"cv2.imencode failed: {str(e)}")  # Print error
                return jsonify({"error": f"Encoding failed: {str(e)}"}), 500

        # Prepare the response
        response = {
            "watermark_detected": len(watermarks) > 0,
            "watermarks": watermarks,
            "heatmap_overlay": heatmap_overlay  
        }

        # Return the response
        return jsonify(response)

    except Exception as e:
        print(f"Unhandled error: {str(e)}")  # Print any error that crashes Flask
        return jsonify({"error": f"Unhandled server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)