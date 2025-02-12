import requests
import base64
import argparse
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def visualize_heatmap():
    parser = argparse.ArgumentParser(description="Send an image to the API for watermark detection and display the heatmap overlay.")
    parser.add_argument("image_path", type=str, help="Path to the test image")
    args = parser.parse_args()
    
    # Send image to API
    with open(args.image_path, "rb") as img_file:
        response = requests.post("http://localhost:5000/predict", files={"image": img_file})
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract the base64-encoded heatmap overlay
        heatmap_base64 = data.get("heatmap_overlay")
        if heatmap_base64:
            # Decode and display the heatmap
            decoded_heatmap = base64.b64decode(heatmap_base64)
            heatmap_image = Image.open(BytesIO(decoded_heatmap))
            
            plt.imshow(heatmap_image)
            plt.axis("off")
            plt.show()
        else:
            print("Error: No heatmap overlay found in the response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}")

if __name__ == "__main__":
    visualize_heatmap()
