Watermark detection and localization using YOLO

The project detects and localizes watermarks in images using a deep learning model. The model is deployed as an API using Flask and containerized with Docker. Kubernetes deployment files are also included.

For running the API, clone the repository and excute the following commands in the terminal:

     cd watermark-detection
     python3 -m venv venv
     source venv/bin/activate  (For Linux/macOS)     OR       venv\Scripts\activate  (For Windows)
     pip install -r requirements.txt
     python scripts/api.py
     curl -X POST -F "image=@path/to/test_image.jpg" http://localhost:5000/predict

The API will return a json reponse in the following format:

    {

    "watermark_detected": true,
    "watermarks": [
        {
            "bbox": [100, 150, 200, 250],
            "confidence": 0.95
        }
    ],
    "heatmap_overlay": "base64_encoded_image_string"

    }

For visualizing the heatmap overlay, run the following command while the API is running:
    
    python scripts/heatmap.py test_image.jpg

Next steps:

Build and run the Docker container:

    docker build -t watermark-detection .
    docker run -p 5000:5000 watermark-detection

Deploy to Kubernetes:

    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    


