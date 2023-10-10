import numpy as np
import cv2
import os
import sys
import json

# Paths
root_dir = "D:/DomainAdap.Neural/Data/"
bdd_img_dir = os.path.join(root_dir, 'BDD_VAL','images') 
img_output_dir = os.path.join(root_dir, 'BDD_VAL', 'resized_images') 
bdd_label_path = os.path.join(root_dir, 'BDD100k', 'bdd100k', 'labels', 'det_val.json')
converted_path = os.path.join(root_dir, 'BDD_VAL', 'labels')  

resized_width = 640
resized_height = 640

with open(bdd_label_path, 'r') as f:
    data = json.load(f)

category_mapping = {
    "car": 0,
    "van": 1,
    "truck": 2,
    "dontcare": 3,
    "bus": 1  
}

# Process Image and Labels
for indexi, entry in enumerate(data):
    img_file = entry["name"]
    img_path = os.path.join(bdd_img_dir, img_file)

    img = cv2.imread(img_path)
    orig_height, orig_width = img.shape[0], img.shape[1]
    resized_img = cv2.resize(img, (resized_width, resized_height))
    resized_img_path = os.path.join(img_output_dir, img_file.replace('.jpg', '.png'))
    cv2.imwrite(resized_img_path, resized_img)

    label_file = img_file.replace('.jpg', '.txt')
    transformed_label_path = os.path.join(converted_path, label_file)

    with open(transformed_label_path, 'w') as final_label:
        for label in entry["labels"]:
            class_str = label["category"]
            if class_str not in category_mapping:
                class_str = 'dontcare'

            orig_left, orig_top, orig_right, orig_bottom = label["box2d"].values()

            resized_left = orig_left * resized_width / orig_width
            resized_top = orig_top * resized_height / orig_height
            resized_right = orig_right * resized_width / orig_width
            resized_bottom = orig_bottom * resized_height / orig_height

            bbox_center_x = ((resized_left + resized_right) / 2.0) / resized_width
            bbox_center_y = ((resized_top + resized_bottom) / 2.0) / resized_height
            bbox_width = (resized_right - resized_left) / resized_width
            bbox_height = (resized_bottom - resized_top) / resized_height

            line_to_write = f"{category_mapping[class_str]} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}\n"
            final_label.write(line_to_write)

        sys.stdout.write(str(int((indexi / len(data)) * 100)) + '% ' + '*******************->' "\r")
        sys.stdout.flush()

print("Label transformations finished!")
