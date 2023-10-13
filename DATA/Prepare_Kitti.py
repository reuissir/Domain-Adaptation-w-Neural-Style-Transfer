import numpy as np
import cv2
import os
import sys

# Defining Paths and Variables
root_dir = "D:/DomainAdap.Neural/Data/kitti"
kitti_img_dir = os.path.join(root_dir, 'data_object_image_2', 'training', 'image_2')
img_output_dir = os.path.join(root_dir, 'KITTI', 'train', 'images')
kitti_label_path = os.path.join(root_dir, 'data_object_label_2', 'training', 'label_2')
converted_path = os.path.join(root_dir, 'KITTI', 'train', 'labels')

resized_width = 640
resized_height = 640
index = 0

## Read kitti.names which contains class names of KITTI
with open('kitti.names','r') as file:
    kitti_names = file.readlines()

# Load Image and Label Filenames
kitti_images = os.listdir(kitti_img_dir)
kitti_labels = os.listdir(kitti_label_path)

kitti_images.sort()
kitti_labels.sort()


# Create Class-to_Index Dictionary
kitti_names_dic_key = ["Car", "Van", "Truck", "DontCare"]
values = range(len(kitti_names_dic_key))

kitti_class_index = dict(zip(kitti_names_dic_key, values))
print(kitti_class_index)

# Create a new file called train.txt
## loop over image filenames + construct the full path + 'img' filename
f = open('train.txt','w')
for img in kitti_images:
    ## write to the train.txt file
    f.write(kitti_img_dir+'/'+img+'\n')
f.close()

# Process Image and Labels
for indexi, (img_file, label_file) in enumerate(zip(kitti_images, kitti_labels)):
    transformed_label_path = os.path.join(converted_path, label_file)    
    img_path = os.path.join(kitti_img_dir, img_file)    
    label_path = os.path.join(kitti_label_path, label_file)
    
    img = cv2.imread(img_path)
    orig_height, orig_width = img.shape[0], img.shape[1]
    resized_img = cv2.resize(img, (resized_width, resized_height))
    resized_img_path = os.path.join(img_output_dir, img_file)
    cv2.imwrite(resized_img_path, resized_img)

    
    with open(label_path, 'r') as kitti_label_contents, open(transformed_label_path, 'w') as final_label:
        for line in kitti_label_contents:
            data = line.split(' ')
            if len(data) == 15:
                class_str = data[0]
                if class_str not in ["Car", "Van", "Truck"]:
                    class_str = 'DontCare'

            # Original KITTI bounding box coordinates
            orig_left = float(data[4])
            orig_top = float(data[5])
            orig_right = float(data[6])
            orig_bottom = float(data[7])

            # Resized Bounding Box Coordinates
            resized_left = orig_left * resized_width / orig_width
            resized_top = orig_top * resized_height / orig_height
            resized_right = orig_right * resized_width / orig_width 
            resized_bottom = orig_bottom * resized_height / orig_height

            # convert bbox coordinates into x, y, w, h + normalization
            bbox_center_x = ((resized_left + resized_right) / 2.0 )/ resized_width
            bbox_center_y = ((resized_top + resized_bottom) / 2.0 )/ resized_height
            bbox_width = (resized_right - resized_left) / resized_width
            bbox_height = (resized_bottom - resized_top) / resized_height

            # Write label contents into YOLO format
            line_to_write = (
                    f"{kitti_class_index[class_str]} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}\n"
                )
                
            final_label.write(line_to_write)
        sys.stdout.write(str(int((indexi / len(kitti_images)) * 100)) + '% ' + '*******************->' "\r")
        sys.stdout.flush()
      
    #cv2.imshow(str(indexi)+' kitti_label_show',kitti_img_totest)    
    #cv2.waitKey()
            
print("Label tranformations finished!")
