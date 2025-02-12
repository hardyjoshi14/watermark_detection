import os
import shutil
from sklearn.model_selection import train_test_split

images_folder = "images/combined_images/"
labels_folder = "labels/"
output_folder = "data/"

split_ratio = 0.8

for data_split in ["train", "val"]:
    os.makedirs(os.path.join(output_folder,data_split,"images"),exist_ok=True)
    os.makedirs(os.path.join(output_folder,data_split,"labels"),exist_ok=True)

image_names = [f for f in os.listdir(images_folder)]

train_data, val_data= train_test_split(image_names, test_size=0.2, random_state=42)

def move_files(image_list, data_split):
    for image_name in image_list:
        image_path = os.path.join(images_folder, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_name)

        dst_image = os.path.join(output_folder, data_split, "images", image_name)
        dst_label = os.path.join(output_folder, data_split, "labels", label_name)

        shutil.copy(image_path, dst_image)
        shutil.copy(label_path, dst_label)

move_files(train_data, "train")
move_files(val_data, "val")