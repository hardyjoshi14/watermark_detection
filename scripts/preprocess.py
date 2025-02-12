import cv2
import numpy as np
import os
import shutil

# Input folders
WATERMARK_FOLDER = "images/r123-watermark/"
NON_WATERMARK_FOLDER = "images/non_watermark/"
ALL_IMAGES_FOLDER = "images/combined_images/"
ALL_LABELS_FOLDER = "labels/"

# Valid image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Create output directories
os.makedirs(ALL_IMAGES_FOLDER, exist_ok=True)
os.makedirs(ALL_LABELS_FOLDER, exist_ok=True)

# Fixed image size for normalization
TARGET_SIZE = (512, 512)

def auto_canny(image, sigma=0.33):
    """Auto-adjust Canny thresholds based on image median."""
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)

def detect_watermark(image_path):
    """Detects watermark and returns bounding box in YOLO format."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return None
    
    image = cv2.resize(image, TARGET_SIZE)  # Normalize image size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # Define region of interest (ROI) for middle 30%
    roi_y_start = int(h * 0.35)
    roi_y_end = int(h * 0.65)
    roi_x_start = int(w * 0.05)
    roi_x_end = int(w * 0.95)

    # Extract the region of interest (middle 30% height, 90% width)
    middle_part = gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Enhance contrast
    equalized = cv2.equalizeHist(middle_part)

    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    # Apply auto-adjusted Canny edge detection
    edges = auto_canny(adaptive_thresh)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        x += roi_x_start
        y += roi_y_start

        # Ignore vertical shapes (ensure horizontal watermark detection)
        if height > width * 0.5:
            continue
        
        # Expand bounding boxes
        expand_x = int(width * 0.3)  
        expand_y = int(height * 0.5)  
        x = max(roi_x_start, x - expand_x)
        y = max(roi_y_start, y - expand_y)
        x2 = min(roi_x_end, x + width + 2 * expand_x)
        y2 = min(roi_y_end, y + height + 2 * expand_y)

        bounding_boxes.append((x, y, x2, y2))

    # If bounding boxes are detected, find the extreme points to create a single large box
    if bounding_boxes:
        x_min = min([box[0] for box in bounding_boxes])
        y_min = min([box[1] for box in bounding_boxes])
        x_max = max([box[2] for box in bounding_boxes])
        y_max = max([box[3] for box in bounding_boxes])

        # Expand slightly to ensure full watermark capture
        padding_x = int((x_max - x_min) * 0.1)  
        padding_y = int((y_max - y_min) * 0.1)  

        x_min = max(roi_x_start, x_min - padding_x)
        y_min = max(roi_y_start, y_min - padding_y)
        x_max = min(roi_x_end, x_max + padding_x)
        y_max = min(roi_y_end, y_max + padding_y)
        
        # Convert to YOLO format (class 0 assumed for watermark)
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
    
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

def process_images(image_folder, is_watermarked=True):
    """Process images, detect watermarks, and save YOLO label files."""
    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith(VALID_EXTENSIONS):
            print(f"Skipping {image_name} (invalid format)")
            continue

        image_path = os.path.join(image_folder, image_name)
        output_image_path = os.path.join(ALL_IMAGES_FOLDER, image_name)
        output_label_path = os.path.join(ALL_LABELS_FOLDER, image_name.rsplit(".", 1)[0] + ".txt")
        
        shutil.copy(image_path, output_image_path)
        
        if is_watermarked:
            bbox = detect_watermark(image_path)
            if bbox:
                with open(output_label_path, "w") as f:
                    f.write(bbox)
            else:
                print(f"No watermark detected in {image_name}")
                open(output_label_path, "w").close()
        else:
            open(output_label_path, "w").close()  # Empty file for non-watermarked images

# Run processing for both folders
process_images(WATERMARK_FOLDER, is_watermarked=True)
process_images(NON_WATERMARK_FOLDER, is_watermarked=False)

print("Labeling complete! Check the 'images' and 'labels' folders.")
